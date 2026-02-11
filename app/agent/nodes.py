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
    EMERGENCY_RESPONSE_PROMPT_TEMPLATE,
    # 코칭 에이전트 프롬프트
    GOAL_SETTER_PROMPT_TEMPLATE,
    GOAL_SETTER_RESET_PROMPT_TEMPLATE,
    GOAL_SETTER_MESSAGE_PROMPT_TEMPLATE,
    GOAL_EVALUATOR_SYSTEM_PROMPT,
    GOAL_EVALUATOR_PROMPT_TEMPLATE,
    COACH_AGENT_PROMPT_TEMPLATE,
    COACH_TOOL_CALL_PROMPT_TEMPLATE,
    EVALUATOR_PROMPT_TEMPLATE,
    COACHING_EVALUATOR_SYSTEM_PROMPT,
    CLOSING_PROMPT_TEMPLATE,
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


@track_node_execution_time("goal_setter")
async def goal_setter_node(state: AgentState) -> AgentState:
    """
    Goal Setter 노드 (코칭 에이전트)
    사용자 발화에서 '해결하고 싶은 문제'를 추출하여 구체적인 목표와 단계를 수립.
    
    [실행 단계 2-Step]
    1. JSON 추출: LLM에게 Goal, Steps를 JSON으로 응답하도록 요청
    2. 메시지 생성(스트리밍): 추출된 Goal, Steps를 바탕으로 사용자 안내 메시지를 스트리밍 생성
    
    재설정 모드: _goal_feedback가 있으면 사용자 피드백을 반영하여 목표를 재수립.
    """
    logger.info("===== 🎯 Goal Setter 노드 실행 =====")
    
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    messages = state.get("messages", [])
    goal_feedback = state.get("_goal_feedback", "")
    prev_goal = state.get("goal", "")
    prev_steps = state.get("coaching_steps", [])
    
    llm = get_generator_llm()
    if not llm:
        logger.error("LLM 클라이언트가 없어 목표 설정을 수행할 수 없습니다.")
        state["response"] = "죄송합니다. 현재 코칭을 시작할 수 없습니다. 잠시 후 다시 시도해주세요."
        state["goal_status"] = "ready"
        return state
    
    try:
        baby_context = get_baby_context_string(baby_info)
        
        # [Step 1] JSON 추출 (Goal + Steps)
        system_prompt = GOAL_SETTER_PROMPT_TEMPLATE.format(
            baby_context=baby_context
        )
        
        # 재설정 모드: 이전 계획과 사용자 피드백을 프롬프트에 포함
        if goal_feedback and prev_goal:
            prev_steps_str = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(prev_steps)]) if prev_steps else "없음"
            system_prompt += GOAL_SETTER_RESET_PROMPT_TEMPLATE.format(
                prev_goal=prev_goal,
                prev_steps_str=prev_steps_str,
                goal_feedback=goal_feedback
            )
            logger.info(f"🔄 목표 재설정 모드 (피드백: {goal_feedback[:50]}...)")
        
        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-5:] if len(clean_messages) > 5 else clean_messages
        recent_history = sanitize_messages_for_llm(recent_history)
        
        # JSON 추출용 호출 (비스트리밍)
        input_messages = [SystemMessage(content=system_prompt)] + recent_history
        response = await llm.ainvoke(input_messages)
        response_text = response.content.strip()
        
        result = parse_json_from_response(response_text)
        
        goal = result.get("goal", "")
        steps = result.get("steps", [])
        
        if not goal or not steps:
            logger.warning("⚠️ 목표/단계 추출 실패, 기본 응답 반환")
            state["response"] = "죄송합니다. 코칭 목표를 설정하는데 어려움이 있었습니다. 어떤 부분이 걱정되시는지 좀 더 자세히 말씀해주시겠어요?"
            state["goal_status"] = "ready"
            state["messages"] = [AIMessage(content=state["response"])]
            return state

        # [Step 2] 메시지 생성 (스트리밍)
        steps_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        
        message_prompt = GOAL_SETTER_MESSAGE_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            goal=goal,
            steps_str=steps_str
        )
        
        # 스트리밍 호출
        msg_response = await llm.ainvoke(
            [SystemMessage(content=message_prompt)], # System prompt만으로 생성 (히스토리는 이미 반영됨)
            config={"tags": ["stream_response"]}
        )
        
        message = msg_response.content.strip()
        
        state["goal"] = goal
        state["coaching_steps"] = steps
        state["current_step_idx"] = 0
        state["goal_status"] = "in_progress"
        state["_goal_feedback"] = None  # 피드백 초기화
        state["_goal_approved"] = None  # 승인 상태 초기화
        state["response"] = message
        state["messages"] = [msg_response]
        
        logger.info(f"✅ 목표 설정 완료: {goal}")
        logger.info(f"✅ 단계 수립: {len(steps)}개 단계")
        logger.info("✅ 안내 메시지 생성 완료 (스트리밍)")

    except Exception as e:
        logger.error(f"목표 설정 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 코칭 준비 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        state["goal_status"] = "ready"
        state["messages"] = [AIMessage(content=state["response"])]
    
    return state


@track_node_execution_time("goal_evaluator")
async def goal_evaluator_node(state: AgentState) -> AgentState:
    """
    Goal Evaluator 노드 (코칭 에이전트)
    Goal Setter가 수립한 목표/계획에 대한 사용자의 승인 여부를 판단.
    
    - approved: coach_agent로 진행 (코칭 시작)
    - modify: goal_setter로 복귀 (사용자 피드백 반영하여 재설정)
    """
    logger.info("===== ✅ Goal Evaluator 노드 실행 =====")
    
    messages = state.get("messages", [])
    goal = state.get("goal", "")
    coaching_steps = state.get("coaching_steps", [])
    
    # 사용자의 최신 메시지 추출
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        logger.warning("⚠️ 사용자 메시지를 찾을 수 없습니다. 기본값(approved) 처리")
        state["_goal_approved"] = True
        return state
    
    llm = get_evaluator_llm()
    if not llm:
        logger.warning("⚠️ 평가 모델이 없어 기본값(approved)으로 처리합니다.")
        state["_goal_approved"] = True
        return state
    
    try:
        all_steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(coaching_steps)])
        
        eval_prompt = GOAL_EVALUATOR_PROMPT_TEMPLATE.format(
            goal=goal,
            all_steps=all_steps,
            user_message=user_message
        )
        
        eval_messages = [
            SystemMessage(content=GOAL_EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt)
        ]
        
        response = await llm.ainvoke(eval_messages)
        response_text = response.content.strip()
        
        result = parse_json_from_response(response_text)
        
        decision = result.get("decision", "approved")
        reason = result.get("reason", "")
        feedback = result.get("feedback", "")
        
        logger.info(f"✅ Goal Evaluator 판단: {decision} (이유: {reason})")
        
        if decision == "approved":
            state["_goal_approved"] = True
            logger.info("👍 목표 승인 → Coach Agent로 진행")
        else:
            state["_goal_approved"] = False
            state["_goal_feedback"] = feedback or user_message
            logger.info(f"✏️ 목표 수정 요청 → Goal Setter로 복귀 (피드백: {feedback[:50]}...)")
        
    except Exception as e:
        logger.error(f"Goal Evaluator 실행 실패: {str(e)}", exc_info=True)
        # 실패 시 기본적으로 approved
        state["_goal_approved"] = True
        logger.info("⚠️ 평가 실패, 기본값(approved)으로 Coach Agent 진행")
    
    return state



@track_node_execution_time("coach_agent")
async def coach_agent_node(state: AgentState) -> AgentState:
    """
    Coach Agent 노드 (코칭 에이전트)
    
    2가지 모드:
    1) Tool 호출 모드: LLM이 tool_calls를 반환 → ToolNode로 라우팅
    2) 응답 생성 모드: ToolMessage(검색 결과)를 문서 컨텍스트로 조합 → 최종 가이드 스트리밍
    
    이 노드 실행 후 (응답 생성 완료 시) interrupt되어 사용자 입력을 대기합니다.
    """
    logger.info("===== 🏋️ Coach Agent 노드 실행 =====")
    
    goal = state.get("goal", "")
    coaching_steps = state.get("coaching_steps", [])
    current_step_idx = state.get("current_step_idx", 0)
    baby_info = state.get("baby_info", {})
    messages = state.get("messages", [])
    
    if not coaching_steps or current_step_idx >= len(coaching_steps):
        logger.warning("⚠️ 유효한 코칭 단계가 없습니다.")
        state["response"] = "코칭 세션에 문제가 발생했습니다."
        state["goal_status"] = "completed"
        return state
    
    current_step = coaching_steps[current_step_idx]
    
    llm = get_generator_llm()
    if not llm:
        state["response"] = "죄송합니다. 현재 가이드를 제공할 수 없습니다."
        return state
    
    # ===== ToolMessage 처리: 이전 Tool 실행 결과에서 문서 추출 =====
    has_tool_results = False
    new_retrieved_docs = []
    new_qna_docs = []
    
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break  # 현재 턴의 사용자 메시지 이전까지만 확인
        if isinstance(msg, AIMessage):
            continue
        if isinstance(msg, ToolMessage):
            has_tool_results = True
            tool_name = getattr(msg, "name", "")
            raw_data = getattr(msg, "artifact", None)
            if not raw_data:
                continue
            
            logger.info(f"🔍 ToolMessage Artifact 추출: {tool_name}")
            if tool_name == "milvus_knowledge_search":
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
    
    if new_retrieved_docs:
        state["_retrieved_docs"] = new_retrieved_docs
        logger.info(f"✅ RAG 문서 State 업데이트: {len(new_retrieved_docs)}개")
    if new_qna_docs:
        state["_qna_docs"] = new_qna_docs
        logger.info(f"✅ QnA 문서 State 업데이트: {len(new_qna_docs)}개")
    
    # ===== 모드 결정 =====
    if not has_tool_results:
        # ---- 모드 1: Tool 호출 모드 (LLM에게 검색 도구를 제공) ----
        logger.info("📡 Tool 호출 모드: LLM이 검색 도구 사용 여부를 결정합니다.")
        
        tools = [milvus_knowledge_search, retrieve_qna]
        model_with_tools = llm.bind_tools(tools)
        
        baby_context = get_baby_context_string(baby_info)
        all_steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(coaching_steps)])
        
        tool_prompt = COACH_TOOL_CALL_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            goal=goal,
            all_steps=all_steps,
            current_step=current_step,
            step_number=current_step_idx + 1,
            total_steps=len(coaching_steps)
        )

        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-5:] if len(clean_messages) > 5 else clean_messages
        recent_history = sanitize_messages_for_llm(recent_history)
        tool_messages = [SystemMessage(content=tool_prompt)] + recent_history
        
        response = await model_with_tools.ainvoke(tool_messages)
        state["messages"] = [response]
        
        # tool_calls가 있으면 ToolNode가 처리, 없으면 바로 응답 생성 모드로 전환
        if response.tool_calls:
            logger.info(f"🔧 Tool 호출 요청: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info("ℹ️ LLM이 Tool 호출 없이 응답 → 바로 응답 생성 모드로 전환")
        
        return state
    
    # ---- 모드 2: 응답 생성 모드 (Tool 결과를 활용하여 가이드 생성) ----
    logger.info("📝 응답 생성 모드: Tool 결과를 활용하여 코칭 가이드를 작성합니다.")
    
    # 문서 컨텍스트 구성
    rag_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", [])
    
    formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
    rag_context = get_docs_context_string(rag_docs)
    
    docs_context = ""
    if formatted_qna:
        docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
    if rag_context:
        docs_context += f"[검색된 문서]\n{rag_context}\n\n"
    if not docs_context:
        docs_context = "관련된 참조 문서가 없습니다. 전문 지식을 바탕으로 가이드해주세요."
    
    # 이전 평가 결과 (재시도 시 다른 방법 제안용)
    eval_context = "없음 (첫 시도)"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'additional_kwargs'):
            feedback = msg.additional_kwargs.get("user_feedback", "")
            if feedback:
                eval_context = f"이전 사용자 피드백: {feedback}"
                break
    
    all_steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(coaching_steps)])
    
    try:
        baby_context = get_baby_context_string(baby_info)
        
        system_prompt = COACH_AGENT_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            goal=goal,
            all_steps=all_steps,
            current_step=current_step,
            step_number=current_step_idx + 1,
            total_steps=len(coaching_steps),
            docs_context=docs_context,
            eval_context=eval_context
        )
        
        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-5:] if len(clean_messages) > 5 else clean_messages
        recent_history = sanitize_messages_for_llm(recent_history)
        
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        state["messages"] = [response]
        
        logger.info(f"✅ Coach Agent 가이드 생성 완료 (단계 {current_step_idx + 1}/{len(coaching_steps)})")
        
    except Exception as e:
        logger.error(f"코치 가이드 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 가이드 생성 중 오류가 발생했습니다."
        state["messages"] = [AIMessage(content=state["response"])]
    
    return state


@track_node_execution_time("coaching_evaluator")
async def coaching_evaluator_node(state: AgentState) -> AgentState:
    """
    Evaluator 노드 (코칭 에이전트)
    사용자의 응답을 분석하여 다음 경로를 결정.
    - success: step_idx + 1 (마지막이면 completed)
    - retry: Coach Agent로 복귀 (다른 방법 제안)
    - stop: paused → Closing
    - chitchat: Coach Agent로 복귀 (질문 답변 후 코칭 유도)
    """
    logger.info("===== 📊 Coaching Evaluator 노드 실행 =====")
    
    goal = state.get("goal", "")
    coaching_steps = state.get("coaching_steps", [])
    current_step_idx = state.get("current_step_idx", 0)
    messages = state.get("messages", [])
    
    current_step = coaching_steps[current_step_idx] if coaching_steps and current_step_idx < len(coaching_steps) else ""
    
    # 사용자의 최신 메시지 추출
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        logger.warning("⚠️ 사용자 메시지를 찾을 수 없습니다.")
        return state
    
    llm = get_evaluator_llm()
    if not llm:
        logger.warning("⚠️ 평가 모델이 없어 기본값(retry)으로 처리합니다.")
        return state
    
    try:
        eval_prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            goal=goal,
            current_step=current_step,
            step_number=current_step_idx + 1,
            total_steps=len(coaching_steps),
            user_message=user_message
        )
        
        eval_messages = [
            SystemMessage(content=COACHING_EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt)
        ]
        
        response = await llm.ainvoke(eval_messages)
        response_text = response.content.strip()
        
        result = parse_json_from_response(response_text)
        
        next_action = result.get("next_action", "retry")
        reason = result.get("reason", "")
        user_feedback = result.get("user_feedback", "")
        
        logger.info(f"✅ Evaluator 판단: {next_action} (이유: {reason})")
        
        if next_action == "success":
            new_idx = current_step_idx + 1
            if new_idx >= len(coaching_steps):
                # 모든 단계 완료
                state["current_step_idx"] = new_idx
                state["goal_status"] = "completed"
                logger.info("🎉 모든 코칭 단계 완료!")
            else:
                # 다음 단계로 이동
                state["current_step_idx"] = new_idx
                logger.info(f"➡️ 다음 단계로 이동: {new_idx + 1}/{len(coaching_steps)}")
                
        elif next_action == "stop":
            state["goal_status"] = "paused"
            logger.info("⏸️ 사용자 요청으로 코칭 중단")
            
        elif next_action == "retry":
            # current_step_idx는 유지, Coach Agent에서 다른 방법 제안
            logger.info("🔄 재시도: Coach Agent에서 다른 방법 제안 예정")
            
        elif next_action == "chitchat":
            # current_step_idx는 유지, Coach Agent에서 질문 답변 후 코칭 유도
            logger.info("💬 잡담 감지: Coach Agent에서 답변 후 코칭으로 복귀 예정")
        
        # 피드백을 상태에 저장 (Coach Agent에서 참조용)
        if user_feedback:
            # 최신 AI 메시지에 피드백 추가
            state["messages"] = [AIMessage(
                content="", 
                additional_kwargs={"user_feedback": user_feedback, "eval_action": next_action}
            )]
            
    except Exception as e:
        logger.error(f"Evaluator 실행 실패: {str(e)}", exc_info=True)
        # 실패 시 기본적으로 retry (Coach Agent로 복귀)
        logger.info("⚠️ 평가 실패, 기본값(retry)으로 Coach Agent 복귀")
    
    return state


@track_node_execution_time("closing")
async def closing_node(state: AgentState) -> AgentState:
    """
    Closing 노드 (코칭 에이전트)
    대화를 종료하고 결과를 정리.
    - completed: 축하 메시지
    - paused: 위로/휴식 권유 메시지
    """
    logger.info("===== 🏁 Closing 노드 실행 =====")
    
    goal = state.get("goal", "")
    coaching_steps = state.get("coaching_steps", [])
    current_step_idx = state.get("current_step_idx", 0)
    goal_status = state.get("goal_status", "completed")
    baby_info = state.get("baby_info", {})
    messages = state.get("messages", [])
    
    llm = get_generator_llm()
    if not llm:
        if goal_status == "completed":
            state["response"] = "🎉 모든 단계를 완료하셨습니다! 정말 대단해요! 앞으로도 아기와 함께 행복한 시간 보내세요."
        else:
            state["response"] = "오늘은 여기까지 할게요. 충분히 잘하고 계세요! 언제든 다시 시작할 수 있어요 💪"
        state["messages"] = [AIMessage(content=state["response"])]
        return state
    
    try:
        baby_context = get_baby_context_string(baby_info)
        all_steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(coaching_steps)])
        
        # 완료한 단계 수 계산
        completed_count = min(current_step_idx, len(coaching_steps))
        
        system_prompt = CLOSING_PROMPT_TEMPLATE.format(
            baby_context=baby_context,
            goal=goal,
            all_steps=all_steps,
            completed_steps=completed_count,
            total_steps=len(coaching_steps),
            status=goal_status
        )
        
        clean_messages = get_clean_messages_for_generation(messages)
        recent_history = clean_messages[-10:] if len(clean_messages) > 10 else clean_messages
        
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt)] + recent_history,
            config={"tags": ["stream_response"]}
        )
        
        state["response"] = response.content.strip()
        state["messages"] = [response]
        
        logger.info(f"✅ Closing 메시지 생성 완료 (상태: {goal_status}, 완료: {completed_count}/{len(coaching_steps)})")
        
    except Exception as e:
        logger.error(f"Closing 메시지 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "코칭을 마무리합니다. 오늘도 수고하셨습니다! 💕"
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


def sanitize_messages_for_llm(messages):
    """
    LLM에 전송하기 전 메시지 리스트를 정제하여 OpenAI API 규칙을 준수하도록 보장.
    
    규칙:
    - ToolMessage는 반드시 직전에 tool_calls가 포함된 AIMessage가 있어야 함
    - tool_calls만 있고 content가 없는 AIMessage는 대응하는 ToolMessage 없이는 무의미
    - 고아(orphaned) ToolMessage와 tool_calls AIMessage를 제거
    
    Args:
        messages: 슬라이싱된 메시지 리스트
    
    Returns:
        OpenAI API에 안전하게 전송할 수 있는 정제된 메시지 리스트
    """
    if not messages:
        return []
    
    result = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, ToolMessage):
            # ToolMessage: 직전 메시지가 tool_calls를 가진 AIMessage이거나 다른 ToolMessage인지 확인
            if result and (
                (isinstance(result[-1], AIMessage) and getattr(result[-1], "tool_calls", None))
                or isinstance(result[-1], ToolMessage)
            ):
                result.append(msg)
            else:
                # 고아 ToolMessage → 스킵
                logger.debug(f"🧹 고아 ToolMessage 제거: {getattr(msg, 'name', 'unknown')}")
            i += 1
            continue
        
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # AIMessage with tool_calls: 다음에 대응하는 ToolMessage가 있는지 확인
            has_tool_response = (i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage))
            if has_tool_response:
                result.append(msg)
            else:
                # 대응하는 ToolMessage 없음 → 스킵
                logger.debug("🧹 대응 ToolMessage 없는 tool_calls AIMessage 제거")
            i += 1
            continue
        
        # HumanMessage, SystemMessage, 일반 AIMessage → 그대로 유지
        result.append(msg)
        i += 1
    
    return result
