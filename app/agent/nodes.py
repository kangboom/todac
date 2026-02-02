"""
노드 함수 (Self-RAG 구조)
"""
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from app.agent.state import AgentState
from app.agent.prompts import (
    DOC_RELEVANCE_PROMPT_TEMPLATE, 
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    AGENT_NODE_PROMPT_TEMPLATE,
    get_baby_context_string,
    get_docs_context_string,
    SIMPLE_RESPONSE_PROMPT_TEMPLATE,
    INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
    ASK_FOR_INFO_PROMPT_TEMPLATE,
    EMERGENCY_RESPONSE_PROMPT_TEMPLATE
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
    
    # missing_info 있으면 provide_missing_info로 설정
    missing_info_data = state.get("_missing_info")
    missing_info = missing_info_data.get("missing_info", []) if missing_info_data else []
        
    if missing_info:
        state["_intent"] = "provide_missing_info"
        return state
    
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


@track_node_execution_time("agent")
async def agent_node(state: AgentState) -> AgentState:
    """
    핵심 에이전트 노드 (Self-RAG)
    - 질문 분석 및 tool 호출 결정
    - Tool 호출이 필요하면 tool 호출
    - 이전 단계의 Tool 실행 결과를 수집하여 State 업데이트
    """
    logger.info("===== 🤖 Agent 노드 실행 =====")

    # 1. ToolMessage 처리 및 State 업데이트
    messages = state.get("messages", [])
    new_retrieved_docs = []
    new_qna_docs = []
    
    # 메시지를 역순으로 확인하며 가장 최근의 ToolMessage들을 분석
    for msg in reversed(messages):
        # HumanMessage가 나오기 전까지의 ToolMessage들만 유효
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage):
            continue 
            
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "")
            raw_data = getattr(msg, "artifact", None)
            
            if not raw_data:
                continue # artifact가 없으면 스킵 (혹은 에러처리)

            logger.info(f"🔍 ToolMessage Artifact 추출: {tool_name}")

            if tool_name == "milvus_knowledge_search":
                # raw_data가 이미 리스트/딕셔너리 객체이므로 바로 순회
                for d in raw_data:
                    try:
                        new_retrieved_docs.append(RagDoc(**d))
                    except Exception as e:
                        logger.error(f"RagDoc 변환 실패: {e}")
            
            elif tool_name == "retrieve_qna":
                for d in raw_data:
                    try:
                        new_qna_docs.append(QnADoc(**d))
                    except Exception as e:
                        logger.error(f"QnADoc 변환 실패: {e}")

    # State 업데이트
    if new_retrieved_docs:
        state["_retrieved_docs"] = new_retrieved_docs
        logger.info(f"✅ RAG 문서 State 업데이트: {len(new_retrieved_docs)}개")
        
    if new_qna_docs:
        state["_qna_docs"] = new_qna_docs
        logger.info(f"✅ QnA 문서 State 업데이트: {len(new_qna_docs)}개")

    # 2. Agent 실행 (LLM 호출)
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    
    llm = get_generator_llm()
    if not llm:
        logger.error("OpenAI 클라이언트가 없어 에이전트를 실행할 수 없습니다.")
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."
        return state
    
    try:
        # bind_tools 사용하여 툴 바인딩
        tools = [
            milvus_knowledge_search,  # RAG 검색 tool
            retrieve_qna,             # QnA 검색 tool
        ]
        model_with_tools = llm.bind_tools(tools)
        
        # 시스템 프롬프트 생성 (아기 정보 포함)
        baby_context = get_baby_context_string(baby_info)
        
        # [수정] system_prompt 인자 제거 (템플릿에 통합됨)
        system_prompt = AGENT_NODE_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            question=question
        )
        
        recent_history = messages[-5:] if len(messages) > 5 else messages
        # 시스템 메시지 추가
        messages_with_system = [SystemMessage(content=system_prompt)] + recent_history
        
        # Agent 실행
        response = await model_with_tools.ainvoke(messages_with_system)
        # 응답을 메시지에 추가
        state["messages"] = [response]
        
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return state


@track_node_execution_time("evaluate")
async def evaluate_node(state: AgentState) -> AgentState:
    """
    Grade Documents Node (Self-RAG)
    검색된 문서의 질문 관련성을 평가
    """
    logger.info("===== 🤖 평가 노드 실행 =====")
    question = state.get("question") or state.get("previous_question")
    
    retrieved_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", []) or []

    if not retrieved_docs and not qna_docs:
        logger.warning("⚠️ 평가할 문서가 없습니다 (RAG 및 QnA 모두 없음).")
        state["_doc_relevance_score"] = 0.0
        state["_doc_relevance_passed"] = False
        return state
    
    llm = get_evaluator_llm()
    if not llm:
        logger.warning("⚠️ 평가 모델이 없어 기본값으로 처리합니다.")
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
        return state
    
    try:
        # 평가 대상 구성
        rag_to_evaluate = retrieved_docs[:5]
        qna_to_evaluate = qna_docs[:3]
        
        docs_summary = ""
        current_idx = 1
        
        # RAG 문서 요약 추가
        for doc in rag_to_evaluate:
            content = getattr(doc, "content", "")
            docs_summary += f"\n문서 {current_idx} (일반 문서):\n{content}\n"
            current_idx += 1
            
        # QnA 문서 요약 추가
        for doc in qna_to_evaluate:
            # Pydantic 모델 접근
            q = getattr(doc, "question", "")
            a = getattr(doc, "answer", "")
            docs_summary += f"\n문서 {current_idx} (QnA):\nQ: {q}\nA: {a}\n"
            current_idx += 1
        
        baby_context = get_baby_context_string(state.get("baby_info", {}))
        evaluation_prompt = DOC_RELEVANCE_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            question=question,
            docs_summary=docs_summary
        )
        
        messages = [
            SystemMessage(content="당신은 문서 관련성을 평가하는 전문가입니다. 객관적이고 정확하게 평가하세요."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        response_text = response.content.strip()
        
        evaluation_result = parse_json_from_response(response_text)
        
        score = float(evaluation_result.get("score", 0.5))
        
        # 관련성 있는 문서 인덱스 추출 (1-based index)
        relevant_indices = evaluation_result.get("relevant_indices", [])
        
        # [추가] 부족한 정보 추출
        missing_info_list = evaluation_result.get("missing_info", [])

        state["_doc_relevance_score"] = max(0.0, min(1.0, score))
        state["_doc_relevance_passed"] = score >= 0.7
        
        # missing_info가 있으면 State에 저장 (이번 턴 기준 덮어쓰기)
        if missing_info_list:
            state["_missing_info"] = {
                "missing_info": missing_info_list,
                "reason": evaluation_result.get("reason", ""),
            }
            logger.info(f"🔍 부족한 정보 식별됨: {missing_info_list}")

        logger.info(f"✅ 관련성 평가 결과: {evaluation_result.get('reason', '')}")
        logger.info(f"✅ 점수={score:.2f}, 통과={state['_doc_relevance_passed']}")
        
        # [수정] RAG와 QnA 분리하여 필터링
        filtered_rag = []
        filtered_qna = []
        
        rag_count = len(rag_to_evaluate)
        
        if relevant_indices:
            for idx in relevant_indices:
                # 1-based index -> 0-based
                real_idx = idx - 1 
                
                if real_idx < rag_count:
                    # RAG 문서 범위
                    filtered_rag.append(rag_to_evaluate[real_idx])
                else:
                    # QnA 문서 범위
                    qna_idx = real_idx - rag_count
                    if qna_idx < len(qna_to_evaluate):
                        filtered_qna.append(qna_to_evaluate[qna_idx])
        
        # 필터링 결과 적용
        logger.info(f"✅ 관련성 필터링 (RAG): {len(retrieved_docs)} -> {len(filtered_rag)}")
        logger.info(f"✅ 관련성 필터링 (QnA): {len(qna_docs)} -> {len(filtered_qna)}")
        
        state["_retrieved_docs"] = filtered_rag
        state["_qna_docs"] = filtered_qna # 필터링된 QnA로 교체
    
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
    
    return state

@track_node_execution_time("generate")
async def generate_node(state: AgentState) -> AgentState:
    """
    Generate Node (Self-RAG)
    검색된 문서를 바탕으로 최종 답변 생성 또는 부족한 정보 요청
    """
    logger.info("--- 🤖 답변 생성 노드 실행 ---")
    question = state.get("question") or state.get("previous_question", "")
    baby_info = state.get("baby_info", {})
    messages = state.get("messages", [])
    
    missing_info_data = state.get("_missing_info")
    is_doc_passed = state.get("_doc_relevance_passed", True)
    
    llm = get_generator_llm()
    if not llm:
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다."
        return state

    prompt = ""
    
    # 1. 정보 부족 시 질문 생성 모드
    if not is_doc_passed and isinstance(missing_info_data, dict):
        logger.info("📝 정보 부족 시 질문 생성 모드(Relevance Failed)")
        missing_info_list = missing_info_data.get("missing_info", [])
        reason = missing_info_data.get("reason", "")
        
        if missing_info_list:
            baby_context = get_baby_context_string(baby_info)
            missing_info_str = ", ".join(missing_info_list)
            
            prompt = ASK_FOR_INFO_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                question=question,
                missing_info=missing_info_str,
                reason=reason
            )
        else:
             logger.warning("missing_info 리스트가 비어있어 일반 답변 모드로 전환")

    # 2. 일반 답변 생성 모드 (정보 부족 모드가 아닐 때)
    if is_doc_passed:
        logger.info("📝 일반 답변 생성 모드(Relevance Passed)")
        retrieved_docs = state.get("_retrieved_docs", [])
        qna_docs = state.get("_qna_docs", [])
        
        # missing_info 문자열 생성
        missing_info_str = "없음"
        if missing_info_data and isinstance(missing_info_data, dict):
             m_list = missing_info_data.get("missing_info", [])
             if m_list:
                 missing_info_str = ", ".join(m_list)
        
        baby_context = get_baby_context_string(baby_info)
        
        # 문서 컨텍스트 구성
        formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
        rag_context = get_docs_context_string(retrieved_docs)
        
        docs_context = ""
        if formatted_qna:
            docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
        if rag_context:
            docs_context += f"[검색된 문서]\n{rag_context}\n\n"
        
        if not docs_context:
            docs_context = "관련된 참조 문서가 없습니다. 당신의 전문 지식으로 답변해주세요."
        
        prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            docs_context=docs_context,
            missing_info=missing_info_str
        )
        
        # 일반 모드에서는 답변 생성 후 missing_info 초기화
        state["_missing_info"] = None

    # 3. 공통 LLM 호출
    try:
        clean_messages = get_clean_messages_for_generation(messages)
        
        # 최근 N개만 참조 (토큰 제한)
        recent_history = clean_messages[-5:] if len(clean_messages) > 5 else clean_messages
            
        response = await llm.ainvoke(
            [SystemMessage(content=prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        state["messages"] = [response]
        
    except Exception as e:
        logger.error(f"답변/질문 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 처리 중 오류가 발생했습니다."
        state["_missing_info"] = None
    
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
