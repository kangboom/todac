"""
노드 함수 (Self-RAG 구조)
"""
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from app.agent.state import AgentState
from app.agent.prompts import (
    SIMPLE_RESPONSE_PROMPT_TEMPLATE,
    INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
    EMERGENCY_RESPONSE_PROMPT_TEMPLATE,
    ASK_SITUATION_PROMPT_TEMPLATE,
    GOAL_OPTIONS_PROMPT_TEMPLATE,
    GROW_RESPONSE_PROMPT_TEMPLATE,
    RESEARCH_AGENT_PROMPT_TEMPLATE,
    EVALUATE_DOCS_PROMPT_TEMPLATE,
    PARSE_GOAL_SELECTION_PROMPT,
    get_baby_context_string,
    get_docs_context_string,
)
from app.agent.tools import milvus_knowledge_search, retrieve_qna
from app.services.qna_service import format_qna_docs
from app.dto.qna import QnADoc
from app.dto.rag import RagDoc
from app.core.llm_factory import get_generator_llm, get_evaluator_llm
from app.agent.utils import parse_json_from_response, track_node_execution_time
import logging

logger = logging.getLogger(__name__)

@track_node_execution_time("intent_classifier")
async def intent_classifier_node(state: AgentState) -> AgentState:
    """
    의도 분류 노드
    질문이 '미숙아 돌봄' 범위인지 판단 + '부족한 정보 제공' 여부 판단
    """
    logger.info("===== 🤖 의도 분류 노드 실행 =====")
    
    question = state.get("question", "")
    
    llm = get_evaluator_llm()
    if not llm:
        logger.warning("평가 모델 없음, 기본값(relevant) 설정")
        state["_intent"] = "irrelevant"
        state["response"] = "죄송합니다. 처리 중 오류가 발생했습니다."
        return state
        
    try:

        messages = state.get("messages", [])
        recent_history = messages[-5:] if len(messages) > 5 else messages
        input_messages = [SystemMessage(content=INTENT_CLASSIFICATION_PROMPT_TEMPLATE)] + recent_history
        
        response = await llm.ainvoke(input_messages)
        response_text = response.content.strip()
        
        result = parse_json_from_response(response_text)
        
        intent = result.get("intent", "relevant")
        reason = result.get("reason", "")
        
        logger.info(f"✅ 의도 분류 결과: {intent} (이유: {reason}) ✅")
        state["_intent"] = intent
        
        # irrelevant인 경우 즉시 답변 생성
        if intent == "irrelevant":
            logger.info("🚫 관련 없는 질문 -> 즉시 거절 응답 생성 🚫")
            try:
                simple_prompt = SIMPLE_RESPONSE_PROMPT_TEMPLATE.format(question=question)
                gen_llm = get_generator_llm()
                if gen_llm:

                    # 스트리밍을 위한 태그 추가
                    resp = await gen_llm.ainvoke(
                        [HumanMessage(content=simple_prompt)],
                        config={"tags": ["stream_response"]}
                    )
                    state["response"] = resp.content.strip()
                    state["messages"] = [resp]
                else:
                    state["response"] = "죄송합니다. 미숙아 및 신생아 돌봄과 관련된 질문만 답변할 수 있습니다."
            except Exception as ex:
                logger.error(f"거절 응답 생성 실패: {str(ex)}")
                state["response"] = "죄송합니다. 처리 중 오류가 발생했습니다."
        
    except Exception as e:
        logger.error(f"의도 분류 실패: {str(e)}")
        state["_intent"] = "relevant"
        
    return state

@track_node_execution_time("emergency_response")
async def emergency_response_node(state: AgentState) -> AgentState:
    """
    [통합] 응급 상황 전용 노드 (검색 + 답변 생성)
    - 별도의 평가 노드 없이 즉시 검색하고 답변을 생성합니다.
    - 검색된 모든 문서를 LLM에게 전달하여 관련성 있는 정보만 선별해 답변하도록 합니다.
    """
    logger.info("===== 🚨 Emergency Response 노드 실행 (Fast-Track) =====")
    
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    state["is_emergency"] = True
    qna_docs = []
    rag_docs = []
    
    try:
        # 1. QnA 검색 (invoke 대신 .func 사용)
        # .func를 호출하면 데코레이터 포장을 벗기고 (content, artifact) 튜플을 직접 받습니다.
        qna_content, qna_artifacts = retrieve_qna.func(query=question)
        
        if qna_artifacts:
            for d in qna_artifacts:
                qna_docs.append(QnADoc(**d))
        
        # 2. Milvus 검색 (.func 사용)
        # 인자를 키워드 아규먼트(kwargs) 형태로 명확히 전달하는 것이 좋습니다.
        milvus_content, milvus_artifacts = milvus_knowledge_search.func(query=question)
        
        if milvus_artifacts:
            for d in milvus_artifacts:
                rag_docs.append(RagDoc(**d))

        # 결과 저장
        state["_qna_docs"] = qna_docs
        state["_retrieved_docs"] = rag_docs
        logger.info(f"🚨 응급 검색 완료: {qna_content} {milvus_content}")
        
    except Exception as e:
        logger.error(f"응급 검색 중 오류(무시하고 진행): {str(e)}")

    # 3. [생성] 응급 답변 생성
    llm = get_generator_llm()
    if not llm:
        state["response"] = "시스템 오류입니다. 즉시 119에 연락하거나 병원을 방문하세요."
        return state
        
    try:
        baby_context = get_baby_context_string(baby_info)
        
        # 문서 컨텍스트 구성
        formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
        rag_context = get_docs_context_string(rag_docs)
        
        docs_context = ""
        if formatted_qna:
            docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
        if rag_context:
            docs_context += f"[검색된 문서]\n{rag_context}\n\n"
        if not docs_context:
            docs_context = "관련된 참조 문서가 없습니다. 의학적 상식에 기반해 답변하세요."

        prompt = EMERGENCY_RESPONSE_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            docs_context=docs_context,
            question=question
        )
        
        # 최근 대화 5개만 참조
        messages = state.get("messages", [])
        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-5:] if len(clean_messages) > 5 else clean_messages
        
        response = await llm.ainvoke(
            [SystemMessage(content=prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        state["messages"] = [response]
        
    except Exception as e:
        logger.error(f"응급 답변 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 오류가 발생했습니다. 즉시 가까운 병원 응급실을 방문하세요."
        
    return state


@track_node_execution_time("ask_situation")
async def ask_situation_node(state: AgentState) -> AgentState:
    """
    Ask Situation 노드 (1단계: 현재 상황 질문)
    - 사용자의 질문을 바탕으로, 현재 상황을 파악하는 공감형 질문을 생성합니다.
    - 이 노드의 출력이 스트리밍되어 사용자에게 전달됩니다.
    - 이후 interrupt로 사용자의 상황 답변을 기다립니다.
    """
    logger.info("===== 🗣️ Ask Situation 노드 실행 =====")
    
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    
    llm = get_generator_llm()
    if not llm:
        default_msg = "더 정확한 도움을 드리기 위해, 현재 아기의 상태나 상황을 조금 더 자세히 말씀해 주시겠어요?"
        state["response"] = default_msg
        state["messages"] = [AIMessage(content=default_msg)]
        return state

    try:
        baby_context = get_baby_context_string(baby_info)
        
        system_prompt = ASK_SITUATION_PROMPT_TEMPLATE.format(
            question=question,
            baby_context=baby_context
        )
        
        messages = state.get("messages", [])
        recent_history = messages[-3:] if len(messages) > 3 else messages
        
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        state["messages"] = [response]
        
        logger.info(f"✅ 상황 질문 생성 완료: {state['response'][:30]}...")
        
    except Exception as e:
        logger.error(f"Ask Situation 생성 실패: {str(e)}", exc_info=True)
        fallback_msg = "더 정확한 조언을 위해 현재 아기 상태를 자세히 알려주시겠어요?"
        state["response"] = fallback_msg
        state["messages"] = [AIMessage(content=fallback_msg)]
    
    return state


@track_node_execution_time("goal_options")
async def goal_options_node(state: AgentState) -> AgentState:
    """
    Goal Options 노드 (2단계: 목표 선택지 제시)
    - interrupt로 받은 사용자의 상황 답변을 활용하여
    - 최초 질문 + 상황 답변을 기반으로 2~3개 목표 선택지를 생성합니다.
    - 이 노드의 출력이 스트리밍되어 사용자에게 전달됩니다.
    - 이후 interrupt로 사용자의 목표 선택을 기다립니다.
    """
    logger.info("===== 🎯 Goal Options 노드 실행 =====")
    
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    messages = state.get("messages", [])
    
    # 1. 사용자 상황 답변 수집 (interrupt 이후 마지막 HumanMessage)
    user_situation = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_situation = msg.content
            break
    
    # user_current_info에도 저장 (이후 단계에서 활용)
    state["user_current_info"] = user_situation
    logger.info(f"📝 사용자 상황 답변: {user_situation[:50]}...")
    
    llm = get_generator_llm()
    if not llm:
        default_msg = "어떤 부분을 가장 먼저 도와드릴까요?\n1. 현재 상황 개선하기\n2. 관련 정보 알아보기"
        state["response"] = default_msg
        state["messages"] = [AIMessage(content=default_msg)]
        state["goal_options"] = ["현재 상황 개선하기", "관련 정보 알아보기"]
        return state

    try:
        baby_context = get_baby_context_string(baby_info)
        
        system_prompt = GOAL_OPTIONS_PROMPT_TEMPLATE.format(
            question=question,
            user_situation=user_situation,
            baby_context=baby_context
        )
        
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt)]
        )
        
        result = parse_json_from_response(response.content.strip())
        
        empathy = result.get("empathy", "")
        options = result.get("options", [])
        closing = result.get("closing", "어떤 걸 먼저 도와드릴까요?")
        
        # 선택지를 state에 저장
        state["goal_options"] = options
        
        # 사용자에게 보여줄 메시지 구성
        display_msg = empathy + "\n\n"
        display_msg += "지금 가장 해결해주고 싶은 게 어떤 건가요?\n\n"
        for i, option in enumerate(options, 1):
            display_msg += f"{i}. {option}\n"
        display_msg += f"\n{closing}"
        
        # 스트리밍 응답으로 전달
        ai_msg = AIMessage(content=display_msg)
        state["response"] = display_msg
        state["messages"] = [ai_msg]
        
        logger.info(f"✅ 목표 선택지 {len(options)}개 생성 완료")
        
    except Exception as e:
        logger.error(f"Goal Options 생성 실패: {str(e)}", exc_info=True)
        fallback_options = ["현재 상황 개선 방법 알아보기", "관련 정보 자세히 알아보기"]
        fallback_msg = "어떤 부분이 가장 궁금하세요?\n\n1. 현재 상황 개선 방법 알아보기\n2. 관련 정보 자세히 알아보기\n\n번호로 골라주시거나, 원하시는 게 따로 있으면 직접 적어주셔도 돼요 😊"
        state["response"] = fallback_msg
        state["messages"] = [AIMessage(content=fallback_msg)]
        state["goal_options"] = fallback_options
    
    return state


@track_node_execution_time("goal_selector")
async def goal_selector_node(state: AgentState) -> AgentState:
    """
    Goal Selector 노드 (목표 선택 파싱)
    - interrupt 이후 사용자의 응답을 분석하여 목표를 설정합니다.
    - Evaluator LLM을 사용하여 번호 선택, 복수 선택, 커스텀 목표를 정확하게 파싱합니다.
    """
    logger.info("===== 🎯 Goal Selector 노드 실행 =====")
    
    messages = state.get("messages", [])
    goal_options = state.get("goal_options", [])
    question = state.get("question", "")
    
    # 1. 사용자의 목표 선택 수집 (두 번째 interrupt 이후 마지막 HumanMessage)
    last_human_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    # 2. Evaluator LLM으로 최종 goal 결정
    selected_goal = last_human_msg  # 기본값: 원문 그대로
    is_relevant = True
    
    if goal_options and last_human_msg:
        try:
            eval_llm = get_evaluator_llm()
            if eval_llm:
                options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(goal_options))
                
                parse_prompt = PARSE_GOAL_SELECTION_PROMPT.format(
                    options_text=options_text,
                    user_response=last_human_msg
                )
                
                parse_response = await eval_llm.ainvoke([SystemMessage(content=parse_prompt)])
                parse_result = parse_json_from_response(parse_response.content.strip())
                
                is_relevant = parse_result.get("is_relevant", True)
                parsed_goal = parse_result.get("goal")
                
                if is_relevant and parsed_goal:
                    selected_goal = parsed_goal
                    logger.info(f"🎯 LLM 파싱 결과: {selected_goal}")
                elif not is_relevant:
                    logger.info(f"🚫 관련 없는 응답 감지: {last_human_msg[:30]}...")
                else:
                    logger.warning("⚠️ 목표 파싱 결과 없음, 원문 사용")
            else:
                logger.warning("⚠️ Evaluator LLM 없음, 원문 사용")
        except Exception as parse_err:
            logger.error(f"목표 선택 파싱 실패 (원문 사용): {parse_err}")
    
    # 3. 관련 없는 응답인 경우 → 되묻기
    if not is_relevant:
        retry_msg = "죄송하지만 지금은 목표를 설정하는 단계예요 😊\n\n"
        # 기존 선택지 다시 보여주기
        for i, opt in enumerate(goal_options, 1):
            retry_msg += f"{i}. {opt}\n"
        retry_msg += "\n번호로 골라주시거나, 원하시는 목표를 직접 적어주세요!"
        
        state["response"] = retry_msg
        state["messages"] = [AIMessage(content=retry_msg)]
        state["_goal_valid"] = False
        logger.info("🔄 목표 재선택 요청 (goal_selector self-loop)")
        return state
    
    state["goal"] = selected_goal
    state["_goal_valid"] = True
    logger.info(f"✅ 최종 설정 목표: {selected_goal}")
    
    # user_current_info가 없으면 question 사용
    if not state.get("user_current_info"):
        state["user_current_info"] = question
    
    return state


@track_node_execution_time("research_agent")
async def research_agent_node(state: AgentState) -> AgentState:
    """
    Research Agent 노드 (Tool Binding 적용)
    - goal_selector_node에서 설정된 목표를 바탕으로
    - LLM이 필요한 도구(QnA, Milvus)를 선택하고 실행합니다.
    """
    logger.info("===== 🕵️ Research Agent 노드 실행 =====")
    
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    user_current_info = state.get("user_current_info", question)
    goal = state.get("goal", "")
    
    # 2. LLM + Tool Binding
    llm = get_generator_llm()
    if not llm:
        logger.error("LLM not found")
        return state

    # 사용할 도구 리스트
    tools = [retrieve_qna, milvus_knowledge_search]
    llm_with_tools = llm.bind_tools(tools)
    
    baby_context = get_baby_context_string(baby_info)
    
    # 프롬프트 구성
    system_prompt = RESEARCH_AGENT_PROMPT_TEMPLATE.format(
        baby_context=baby_context,
        question=question,
        user_current_info=user_current_info,
        goal=goal,
    )
    
    try:
        # LLM 호출
        response = await llm_with_tools.ainvoke(
            [SystemMessage(content=system_prompt)],
            config={"tags": ["tool_selection"]}
        )
        
        qna_docs = []
        rag_docs = []
        
        # 3. 도구 실행 (Manual Execution to capture artifacts)
        if response.tool_calls:
            logger.info(f"🛠️ 도구 호출 감지: {len(response.tool_calls)}개")
            
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                logger.info(f"  -> Executing {name} with args: {args}")
                
                try:
                    if name == "retrieve_qna":
                        # .func()를 사용하여 content와 artifacts(metadata)를 모두 가져옴
                        content, artifacts = retrieve_qna.func(**args)
                        if artifacts:
                            for d in artifacts:
                                qna_docs.append(QnADoc(**d))
                                
                    elif name == "milvus_knowledge_search":
                        content, artifacts = milvus_knowledge_search.func(**args)
                        if artifacts:
                            for d in artifacts:
                                rag_docs.append(RagDoc(**d))
                except Exception as tool_err:
                    logger.error(f"❌ 도구 실행 실패 ({name}): {tool_err}")
                    
        else:
            logger.info("⚠️ 도구 호출 없음: LLM이 검색이 필요없다고 판단하거나 실패함.")
            
        # 결과 저장
        state["_qna_docs"] = qna_docs
        state["_retrieved_docs"] = rag_docs
        logger.info(f"✅ Research 완료: QnA {len(qna_docs)}개, Docs {len(rag_docs)}개")
        
    except Exception as e:
        logger.error(f"Research Agent 실패: {e}", exc_info=True)
        
    return state


@track_node_execution_time("evaluate_docs")
async def evaluate_docs_node(state: AgentState) -> AgentState:
    """
    Evaluate Docs 노드
    - 검색된 문서들을 LLM으로 평가하여, 관련 있는 문서의 인덱스만 선별합니다.
    - 선별된 인덱스에 해당하는 원본 문서만 state에 남깁니다.
    """
    logger.info("===== 🧐 Evaluate Docs 노드 실행 =====")
    
    question = state.get("question", "")
    goal = state.get("goal", "")
    user_current_info = state.get("user_current_info", "")
    baby_info = state.get("baby_info", {})
    
    rag_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", [])
    
    # 1. 문서가 하나도 없으면 바로 통과
    if not rag_docs and not qna_docs:
        logger.info("ℹ️ 검색된 문서 없음 -> 평가 생략")
        return state

    llm = get_evaluator_llm()
    if not llm:
        logger.warning("평가 모델 없음 -> 모든 문서 그대로 사용")
        return state

    try:
        baby_context = get_baby_context_string(baby_info)
        
        # 2. QnA 문서 목록 텍스트 생성 (번호 포함)
        qna_docs_list = "없음"
        if qna_docs:
            lines = []
            for i, doc in enumerate(qna_docs):
                q = doc.get("question", "") if isinstance(doc, dict) else getattr(doc, "question", "")
                a = doc.get("answer", "") if isinstance(doc, dict) else getattr(doc, "answer", "")
                lines.append(f"[{i}] Q: {q}\n    A: {a[:200]}...")
            qna_docs_list = "\n".join(lines)
        
        # 3. RAG 문서 목록 텍스트 생성 (번호 포함)
        rag_docs_list = "없음"
        if rag_docs:
            lines = []
            for i, doc in enumerate(rag_docs):
                content = doc.get("content", "") if isinstance(doc, dict) else getattr(doc, "content", "")
                filename = doc.get("filename", "N/A") if isinstance(doc, dict) else getattr(doc, "filename", "N/A")
                lines.append(f"[{i}] (출처: {filename}) {content[:300]}...")
            rag_docs_list = "\n".join(lines)
        
        # 4. 프롬프트 구성 및 LLM 호출
        prompt = EVALUATE_DOCS_PROMPT_TEMPLATE.format(
            question=question,
            goal=goal,
            user_current_info=user_current_info,
            baby_context=baby_context,
            qna_docs_list=qna_docs_list,
            rag_docs_list=rag_docs_list
        )
        
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        result = parse_json_from_response(response.content.strip())
        
        # 5. 인덱스 기반 필터링
        relevant_qna_indices = result.get("relevant_qna_indices", [])
        relevant_rag_indices = result.get("relevant_rag_indices", [])
        reason = result.get("reason", "")
        
        # QnA 필터링
        if qna_docs and relevant_qna_indices:
            filtered_qna = [qna_docs[i] for i in relevant_qna_indices if i < len(qna_docs)]
            state["_qna_docs"] = filtered_qna
            logger.info(f"📋 QnA 필터링: {len(qna_docs)} -> {len(filtered_qna)}개")
        elif qna_docs and not relevant_qna_indices:
            state["_qna_docs"] = []
            logger.info(f"📋 QnA 필터링: {len(qna_docs)} -> 0개 (관련 없음)")
        
        # RAG 필터링
        if rag_docs and relevant_rag_indices:
            filtered_rag = [rag_docs[i] for i in relevant_rag_indices if i < len(rag_docs)]
            state["_retrieved_docs"] = filtered_rag
            logger.info(f"📄 RAG 필터링: {len(rag_docs)} -> {len(filtered_rag)}개")
        elif rag_docs and not relevant_rag_indices:
            state["_retrieved_docs"] = []
            logger.info(f"📄 RAG 필터링: {len(rag_docs)} -> 0개 (관련 없음)")
        
        logger.info(f"✅ 문서 평가 완료 (사유: {reason})")
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        # 실패 시 원본 그대로 유지
        
    return state


@track_node_execution_time("response_node")
async def grow_response_node(state: AgentState) -> AgentState:
    """
    Response Node (GROW 모델 적용)
    - 수집된 정보(Baby Info, User Reality, Goal, Docs)를 바탕으로
    - GROW 모델 프롬프트에 따라 최종 답변을 생성합니다.
    """
    logger.info("===== 🌱 GROW Response 노드 실행 =====")
    question = state.get("question", "")
    goal = state.get("goal", "")
    user_current_info = state.get("user_current_info", "")
    baby_info = state.get("baby_info", {})
    
    # 평가 노드에서 필터링된 원본 문서를 사용
    rag_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", [])
    
    # 문서 컨텍스트 구성
    formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
    rag_context = get_docs_context_string(rag_docs)
    
    docs_context = ""
    if formatted_qna: docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
    if rag_context: docs_context += f"[검색된 문서]\n{rag_context}\n\n"
    if not docs_context: docs_context = "관련 문서 없음 (의학적 상식에 기반하여 답변)"
    
    llm = get_generator_llm()
    if not llm:
        state["response"] = "죄송합니다. 답변을 생성할 수 없습니다."
        return state
        
    try:
        # 문서 컨텍스트 구성 (이미 위에서 docs_context로 준비됨)
        
        baby_context = get_baby_context_string(baby_info)
        
        # GROW 프롬프트 적용
        system_prompt = GROW_RESPONSE_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            user_current_info=user_current_info,
            docs_context=docs_context,
            goal=goal,
            question=question
        )
        
        messages = state.get("messages", [])
        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-5:] # 최근 대화 일부 포함
        
        # 시스템 프롬프트 + 히스토리 -> 답변 생성
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        # 답변을 마지막 메시지로 추가
        state["messages"] = [response]
        
        logger.info("✅ GROW 답변 생성 완료")
        
    except Exception as e:
        logger.error(f"GROW 답변 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["messages"] = [AIMessage(content=state["response"])]
        
    return state


def get_clean_messages_for_generation(messages):
    """
    메시지 히스토리에서 최근 HumanMessage까지만 남기고 그 이후의 Agent 활동 로그는 제거
    
    Args:
        messages: 메시지 리스트
    
    Returns:
        정리된 메시지 리스트
    """
    if not messages:
        return []
    
    # 1. 뒤에서부터 탐색하여 '가장 최근의 HumanMessage' 인덱스 찾기
    last_human_index = -1
    for i, msg in enumerate(reversed(messages)):
        if isinstance(msg, HumanMessage):
            # reversed 상태이므로 원래 인덱스로 변환
            last_human_index = len(messages) - 1 - i
            break
            
    # 2. HumanMessage가 없다면? (예외처리)
    if last_human_index == -1:
        return messages[-10:]  # 그냥 최근꺼 반환
        
    # 3. [핵심] 마지막 질문까지만 남기고, 그 뒤의 Agent 활동 로그는 전부 삭제
    clean_history = messages[:last_human_index + 1]
    
    return clean_history



