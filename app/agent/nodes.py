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
    EMERGENCY_PROMPT_TEMPLATE, 
    SIMPLE_RESPONSE_PROMPT_TEMPLATE,
    INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
    ANALYZE_MISSING_INFO_PROMPT_TEMPLATE,
    CREATE_QUERY_FROM_INFO_PROMPT_TEMPLATE,
    ASK_FOR_INFO_PROMPT_TEMPLATE # [추가]
)
from app.agent.tools import milvus_knowledge_search, report_emergency, retrieve_qna
from app.services.qna_service import format_qna_docs
from app.dto.qna import QnADoc
from app.dto.rag import RagDoc
from app.core.llm_factory import get_generator_llm, get_evaluator_llm
from app.agent.utils import parse_tool_result, parse_json_from_response
import logging

logger = logging.getLogger(__name__)


async def intent_classifier_node(state: AgentState) -> AgentState:
    """
    의도 분류 노드
    질문이 '미숙아 돌봄' 범위인지 판단 + '부족한 정보 제공' 여부 판단
    """
    logger.info("--- 🤖 의도 분류 노드 실행 ---")
    
    # missing_info 있으면 provide_missing_info로 설정
    missing_info_data = state.get("_missing_info")
    missing_info = missing_info_data.get("missing_info", []) if missing_info_data else []
        
    if missing_info:
        logger.info(f"✅ 부족한 정보 요청 상태(missing_info 존재) -> 강제로 provide_missing_info로 설정")
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
        prompt = INTENT_CLASSIFICATION_PROMPT_TEMPLATE.format(
            question=question
        )
        messages = [HumanMessage(content=prompt)]
        
        response = await llm.ainvoke(messages)
        response_text = response.content.strip()
        
        result = parse_json_from_response(response_text)
        
        intent = result.get("intent", "relevant")
        reason = result.get("reason", "")
        
        logger.info(f"의도 분류 결과: {intent} (이유: {reason})")
        state["_intent"] = intent
        
        # irrelevant인 경우 즉시 답변 생성
        if intent == "irrelevant":
            logger.info("🚫 관련 없는 질문 -> 즉시 거절 응답 생성")
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


async def create_query_from_info_node(state: AgentState) -> AgentState:
    """
    Create Query From Info Node
    부족했던 정보가 제공되면, 이를 원본 질문과 결합하여 새로운 검색 질문을 생성
    """
    logger.info("--- 🤖 질문 재구성 노드 실행행 ---")
    
    # missing_info 데이터 구조 처리 (타입 안전하게 처리)
    missing_info_data = state.get("_missing_info") or {}
    missing_info = missing_info_data.get("missing_info", []) if isinstance(missing_info_data, dict) else []
    saved_previous_question = missing_info_data.get("pending_question", "") if isinstance(missing_info_data, dict) else ""
        
    # 저장된 원본 질문이 있으면 사용, 없으면 현재 state의 원본 질문(현재 턴 입력) 사용
    previous_question = saved_previous_question if saved_previous_question else state.get("previous_question", "")
    
    logger.info(f"이전 질문: {previous_question}")
    user_response = state.get("question", "") # 현재 턴의 사용자 입력(정보 제공)
    
    llm = get_generator_llm()
    if not llm:
        return state
    
    missing_info_text = ", ".join(missing_info) if missing_info else ""
        
    prompt = CREATE_QUERY_FROM_INFO_PROMPT_TEMPLATE.format(
        previous_question=previous_question,
        missing_info=missing_info_text,
        user_response=user_response
    )
    
    try:
        # [Async] invoke -> ainvoke
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        new_query = response.content.strip()
        
        logger.info(f"새로운 검색 질문 생성: '{new_query}'")
        
        # 생성된 질문으로 question 업데이트
        state["question"] = new_query
        
        # missing_info 초기화 (해결됨)
        state["_missing_info"] = None
        
        # 재시도 플래그 설정 (무한 루프 방지)
        state["is_retry"] = True
        
    except Exception as e:
        logger.error(f"질문 생성 실패: {e}")
        # 실패 시 원본 질문과 사용자 입력을 단순 결합
        state["question"] = f"{previous_question} {user_response}"
        
    return state


async def agent_node(state: AgentState) -> AgentState:
    """
    핵심 에이전트 노드 (Self-RAG)
    - 질문 분석 및 tool 호출 결정
    - Tool 호출이 필요하면 tool 호출
    - 이전 단계의 Tool 실행 결과를 수집하여 State 업데이트
    """
    logger.info("--- 🤖 Agent 노드 실행 ---")

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
            content = msg.content
            
            logger.info(f"🔍 ToolMessage 분석: {tool_name}")
            
            if tool_name == "milvus_knowledge_search":
                docs = parse_tool_result(content)
                if docs:
                    for d in docs:
                        try:
                            # 딕셔너리를 RagDoc 객체로 변환
                            rag_doc = RagDoc(**d)
                            new_retrieved_docs.append(rag_doc)
                        except Exception as e:
                            logger.error(f"RagDoc 변환 실패: {e}")
                
            elif tool_name == "retrieve_qna":
                docs = parse_tool_result(content)
                if docs:
                    for d in docs:
                        try:
                            qna_doc = QnADoc(**d)
                            new_qna_docs.append(qna_doc)
                        except Exception as e:
                            logger.error(f"QnADoc 변환 실패: {e}")

            elif tool_name == "report_emergency":
                logger.info("  -> 응급 상황 보고 확인")
                state["is_emergency"] = True

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
            report_emergency,         # 응급 상태 보고 tool
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
        
        # 시스템 메시지 추가
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
        
        # Agent 실행
        response = await model_with_tools.ainvoke(messages_with_system)

        # 응답을 메시지에 추가
        state["messages"] = [response]
        
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        state["is_emergency"] = False
    
    return state


async def evaluate_node(state: AgentState) -> AgentState:
    """
    Grade Documents Node (Self-RAG)
    검색된 문서의 질문 관련성을 평가
    """
    logger.info("--- 🤖 평가 노드 실행 ---")
    question = state.get("question") or state.get("previous_question")
    
    retrieved_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", []) or []

    if not retrieved_docs and not qna_docs:
        logger.warning(f"⚠️ 평가할 문서가 없습니다 (RAG 및 QnA 모두 없음).")
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
        
        evaluation_prompt = DOC_RELEVANCE_PROMPT_TEMPLATE.format(
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

        state["_doc_relevance_score"] = max(0.0, min(1.0, score))
        state["_doc_relevance_passed"] = score >= 0.7
        logger.info(f"✅ 관련성 평가 결과: {evaluation_result.get('reason', '')}")
        logger.info(f"점수={score:.2f}, 통과={state['_doc_relevance_passed']}")
        
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
        logger.info(f"관련성 필터링 (RAG): {len(retrieved_docs)} -> {len(filtered_rag)}")
        logger.info(f"관련성 필터링 (QnA): {len(qna_docs)} -> {len(filtered_qna)}")
        
        state["_retrieved_docs"] = filtered_rag
        state["_qna_docs"] = filtered_qna # 필터링된 QnA로 교체
    
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
    
    return state


async def analyze_missing_info_node(state: AgentState) -> AgentState:
    """
    Analyze Missing Info Node
    문서가 불충분할 때 사용자에게 필요한 정보와 그 이유를 분석
    """
    logger.info("--- 🤖 부족한 정보 분석 노드 실행 ---")
    question = state.get("question") or state.get("previous_question", "")
    baby_info = state.get("baby_info", {})
    
    llm = get_evaluator_llm()
    if not llm:
        state["response"] = "죄송합니다. 현재 정보를 찾을 수 없어 답변이 어렵습니다."
        return state
        
    baby_context = get_baby_context_string(baby_info)
    
    prompt = ANALYZE_MISSING_INFO_PROMPT_TEMPLATE.format(
        question=question,
        baby_context=baby_context
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # JSON 파싱
        result = parse_json_from_response(response_text)
        
        # 누락 정보 및 이유 추출
        missing_info_list = result.get("missing_info", [])
        reason = result.get("reason", "")
        
        logger.info(f"부족한 정보 분석 완료")
        logger.info(f"누락 정보: {missing_info_list}, 이유: {reason}")
        
        # missing_info 필드에 저장
        state["_missing_info"] = {
            "missing_info": missing_info_list,
            "reason": reason,
            "pending_question": question
        }
        
    except Exception as e:
        logger.error(f"부족한 정보 분석 실패: {e}")
        state["_missing_info"] = None
        state["response"] = "죄송합니다. 분석 중 오류가 발생했습니다."
        
    return state


async def generate_node(state: AgentState) -> AgentState:
    """
    Generate Node (Self-RAG)
    검색된 문서를 바탕으로 최종 답변 생성 또는 부족한 정보 요청
    """
    logger.info("--- 🤖 답변 생성 노드 실행 ---")
    question = state.get("question") or state.get("previous_question", "")
    baby_info = state.get("baby_info", {})
    
    missing_info_data = state.get("_missing_info")
    
    # 1. 정보 부족 시 질문 생성 모드
    if missing_info_data and isinstance(missing_info_data, dict):
        logger.info("📝 Missing Info Question Generation Mode")
        
        missing_info_list = missing_info_data.get("missing_info", [])
        reason = missing_info_data.get("reason", "")
        
        if not missing_info_list:
            # 리스트가 비었다면 그냥 일반 답변 모드로 진행 (혹은 에러 처리)
            logger.warning("missing_info 리스트가 비어있어 일반 답변 모드로 전환")
            missing_info_data = None
        else:
            baby_context = get_baby_context_string(baby_info)
            missing_info_str = ", ".join(missing_info_list)
            
            prompt = ASK_FOR_INFO_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                question=question,
                missing_info=missing_info_str,
                reason=reason
            )
            
            try:
                llm = get_generator_llm()
                if not llm:
                    state["response"] = "죄송합니다. 모델을 불러올 수 없습니다."
                    return state
                    
                response = await llm.ainvoke(
                    [SystemMessage(content=prompt)],
                    config={"tags": ["stream_response"]}
                )
                generated_response = response.content.strip()
                state["response"] = generated_response
                state["messages"] = [response]
                state["is_emergency"] = False
                state["_retrieved_docs"] = []
                state["_qna_docs"] = []
                
                return state
                
            except Exception as e:
                logger.error(f"질문 생성 실패: {str(e)}", exc_info=True)
                state["response"] = "죄송합니다. 질문 생성 중 오류가 발생했습니다."
                return state

    # 2. 일반 답변 생성 모드
    
    retrieved_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", [])

    llm = get_generator_llm()
    if not llm:
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다."
        return state
    
    try:
        baby_context = get_baby_context_string(baby_info)
        
        # QnA와 RAG 문서 컨텍스트 합치기 (공통 로직)
        formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
        rag_context = get_docs_context_string(retrieved_docs)
        
        docs_context = ""
        if formatted_qna:
            docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
        if rag_context:
            docs_context += f"[검색된 문서]\n{rag_context}\n\n"
        
        if not docs_context:
            docs_context = "관련된 참조 문서가 없습니다. 당신의 전문 지식으로 답변해주세요."
        
        # 응급/일반 모드에 따라 프롬프트 선택
        prompt = ""
        
        if state.get("is_emergency"):
            logger.info("🚨 Emergency Mode: 응급 프롬프트 적용")
            prompt = EMERGENCY_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                full_context=docs_context,
                previous_question=question
            )
        else:
            prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                docs_context=docs_context
            )
        
        response = await llm.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=question)
            ],
            config={"tags": ["stream_response"]}
        )

        generated_response = response.content.strip()
        state["response"] = generated_response
        state["is_emergency"] = False
        
        # [추가] 답변이 생성되었으므로, 부족한 정보 요청 상태 초기화
        state["_missing_info"] = None 
        
        # 메시지에 추가
        state["messages"] = [response]
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["is_emergency"] = False
        state["_missing_info"] = None
    
    return state
