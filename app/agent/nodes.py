"""
노드 함수 (Self-RAG 구조)
"""
from langchain_openai import ChatOpenAI
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
    CREATE_QUERY_FROM_INFO_PROMPT_TEMPLATE # [추가]
)
from app.agent.tools import milvus_knowledge_search, report_emergency, retrieve_qna
from app.services.qna_service import format_qna_docs
from app.dto.qna import QnADoc
from app.dto.rag import RagDoc
from app.core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

# LangChain OpenAI 클라이언트 (에이전트용)
agent_chat_model = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL_GENERATION,
    temperature=0.7,
    max_tokens=1000
) if settings.OPENAI_API_KEY else None

# LangChain OpenAI 클라이언트 (평가용 - 낮은 temperature)
evaluation_chat_model = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL_GENERATION,
    temperature=0.1,  # 평가는 낮은 temperature 사용
    max_tokens=600
) if settings.OPENAI_API_KEY else None


def _parse_tool_result(content: str | list) -> list:
    """ToolMessage의 content를 파싱하여 리스트로 반환"""
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        try:
            # JSON 문자열 파싱
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            return []
    return []


def _parse_json_from_response(text: str) -> dict:
    """LLM 응답 텍스트에서 JSON을 추출하여 파싱"""
    try:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"JSON 파싱 실패: {text[:50]}...")
        return {}
    except Exception as e:
        logger.error(f"JSON 추출 중 오류: {str(e)}")
        return {}


async def intent_classifier_node(state: AgentState) -> AgentState:
    """
    의도 분류 노드
    질문이 '미숙아 돌봄' 범위인지 판단 + '부족한 정보 제공' 여부 판단
    """
    logger.info("--- [NODE] Intent Classification Start ---")
    question = state.get("question", "") or state.get("previous_question", "")
    
    # missing_info 데이터 구조 처리 (Dict or List or None)
    missing_info_data = state.get("_missing_info")
    missing_info = []
    
    if isinstance(missing_info_data, dict):
        missing_info = missing_info_data.get("missing_info", [])
    elif isinstance(missing_info_data, list):
        missing_info = missing_info_data
    
    if not state.get("previous_question"):
        state["previous_question"] = question
        
    # [추가] missing_info가 있다면 무조건 provide_missing_info로 설정 (LLM 판단 생략)
    if missing_info:
        logger.info(f"✅ 부족한 정보 요청 상태(missing_info 존재) -> 강제로 provide_missing_info로 설정")
        state["_intent"] = "provide_missing_info"
        return state

    if not evaluation_chat_model:
        logger.warning("평가 모델 없음, 기본값(relevant) 설정")
        state["_intent"] = "relevant"
        return state
        
    try:
        # missing_info가 있으면 프롬프트에 포함, 없으면 "없음"으로 처리
        missing_info_text = ", ".join(missing_info) if missing_info else "없음"
        
        prompt = INTENT_CLASSIFICATION_PROMPT_TEMPLATE.format(
            question=question
        )
        messages = [HumanMessage(content=prompt)]
        
        # [Async] invoke -> ainvoke
        response = await evaluation_chat_model.ainvoke(messages)
        response_text = response.content.strip()
        
        # [수정] 공통 함수 사용
        result = _parse_json_from_response(response_text)
        
        intent = result.get("intent", "relevant")
        reason = result.get("reason", "")
        
        logger.info(f"의도 분류 결과: {intent} (이유: {reason})")
        state["_intent"] = intent
        
        # irrelevant인 경우 즉시 답변 생성
        if intent == "irrelevant":
            logger.info("🚫 관련 없는 질문 -> 즉시 거절 응답 생성")
            try:
                simple_prompt = SIMPLE_RESPONSE_PROMPT_TEMPLATE.format(question=question)
                # agent_chat_model을 사용하여 자연스러운 답변 생성
                if agent_chat_model:
                    # [Async] invoke -> ainvoke
                    resp = await agent_chat_model.ainvoke([HumanMessage(content=simple_prompt)])
                    state["response"] = resp.content.strip()
                    state["messages"] = [resp]
                else:
                    state["response"] = "죄송합니다. 미숙아 및 신생아 돌봄과 관련된 질문만 답변할 수 있습니다."
            except Exception as ex:
                logger.error(f"거절 응답 생성 실패: {str(ex)}")
                state["response"] = "죄송합니다. 처리 중 오류가 발생했습니다."
        
    except Exception as e:
        logger.error(f"의도 분류 실패: {str(e)}")
        state["_intent"] = "relevant" # 실패 시 안전하게 relevant로 처리
        
    return state


async def create_query_from_info_node(state: AgentState) -> AgentState:
    """
    Create Query From Info Node
    부족했던 정보가 제공되면, 이를 원본 질문과 결합하여 새로운 검색 질문을 생성
    """
    logger.info("--- [NODE] Create Query From Info Start ---")
    
    # missing_info 데이터 구조 처리
    missing_info_data = state.get("_missing_info") or {}
    missing_info = []
    saved_previous_question = ""
    
    if isinstance(missing_info_data, dict):
        missing_info = missing_info_data.get("missing_info", [])
        saved_previous_question = missing_info_data.get("pending_question", "")
    elif isinstance(missing_info_data, list):
        missing_info = missing_info_data
        
    # 저장된 원본 질문이 있으면 사용, 없으면 현재 state의 원본 질문(현재 턴 입력) 사용
    previous_question = saved_previous_question if saved_previous_question else state.get("previous_question", "")
    
    logger.info(f"❓ previous_question: {previous_question}")
    user_response = state.get("question", "") # 현재 턴의 사용자 입력(정보 제공)
    
    if not agent_chat_model:
        return state
    
    missing_info_text = ", ".join(missing_info) if missing_info else ""
        
    prompt = CREATE_QUERY_FROM_INFO_PROMPT_TEMPLATE.format(
        previous_question=previous_question,
        missing_info=missing_info_text,
        user_response=user_response
    )
    
    try:
        # [Async] invoke -> ainvoke
        response = await agent_chat_model.ainvoke([HumanMessage(content=prompt)])
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
    - Tool 호출이 필요하면 tool 호출, 없으면 직접 답변
    - 이전 단계의 Tool 실행 결과를 수집하여 State 업데이트
    """
    logger.info("--- [NODE] Agent Analysis Start ---")

    # 1. ToolMessage 처리 및 State 업데이트
    messages = state.get("messages", [])
    new_retrieved_docs = []
    new_qna_docs = []
    
    # 메시지를 역순으로 확인하며 가장 최근의 ToolMessage들을 분석
    # (HumanMessage가 나오기 전까지의 ToolMessage들만 유효)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage):
            continue 
            
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "")
            content = msg.content
            
            logger.info(f"🔍 ToolMessage 분석: {tool_name}")
            
            if tool_name == "milvus_knowledge_search":
                docs = _parse_tool_result(content)
                if docs:
                    for d in docs:
                        try:
                            # 딕셔너리를 RagDoc 객체로 변환
                            rag_doc = RagDoc(**d)
                            new_retrieved_docs.append(rag_doc)
                        except Exception as e:
                            logger.error(f"RagDoc 변환 실패: {e}")
                    logger.info(f"  -> RAG 문서 {len(docs)}개 발견")
                
            elif tool_name == "retrieve_qna":
                docs = _parse_tool_result(content)
                if docs:
                    for d in docs:
                        try:
                            # 딕셔너리를 QnADoc 객체로 변환 (필드명 매핑 주의)
                            # Tool에서 반환하는 JSON 키와 QnADoc 필드가 일치해야 함
                            qna_doc = QnADoc(**d)
                            new_qna_docs.append(qna_doc)
                        except Exception as e:
                            logger.error(f"QnADoc 변환 실패: {e}")
                    logger.info(f"  -> QnA 문서 {len(docs)}개 발견")

            elif tool_name == "report_emergency":
                logger.info("  -> 응급 상황 보고 확인")
                state["is_emergency"] = True

    # State 업데이트 (새로운 결과가 있을 때만 덮어쓰기)
    if new_retrieved_docs:
        state["_retrieved_docs"] = new_retrieved_docs
        logger.info(f"✅ RAG 문서 State 업데이트: {len(new_retrieved_docs)}개")
        
    if new_qna_docs:
        state["_qna_docs"] = new_qna_docs
        logger.info(f"✅ QnA 문서 State 업데이트: {len(new_qna_docs)}개")

    # 2. Agent 실행 (LLM 호출)
    question = state.get("question", "")
    baby_info = state.get("baby_info", {})
    
    if not agent_chat_model:
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
        model_with_tools = agent_chat_model.bind_tools(tools)
        
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
        # [Async] invoke -> ainvoke
        response = await model_with_tools.ainvoke(messages_with_system)
        
        # 툴 호출 확인하여 is_emergency 플래그 설정 (현재 턴의 호출 확인)
        # 이미 위에서 이전 턴의 report_emergency는 처리했지만, 이번 턴에 또 부를 수도 있음
        
        has_tool_calls = False
        if hasattr(response, 'tool_calls') and response.tool_calls:
            has_tool_calls = True
        elif isinstance(response, dict) and response.get('tool_calls'):
            has_tool_calls = True
            
        if has_tool_calls:
            tool_calls = getattr(response, 'tool_calls', []) or response.get('tool_calls', [])
            
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                logger.info(f"🛠️ Tool Call 감지: {tool_name}")
                
                # 응급 툴이 호출되면 플래그 True 설정
                if tool_name == 'report_emergency':
                    logger.info(f"🚨 응급 툴 호출 감지 -> 응급 모드 활성화")
                    state["is_emergency"] = True
            
            tool_calls_count = len(tool_calls)
            logger.info(f"Tool 호출 결정: {tool_calls_count}개 tool 호출")
            
        else:
            # Tool 호출이 없으면 직접 답변 (AIMessage content 사용)
            # 하지만 여기서 답변을 확정하지 않고, evaluate_node로 넘길 수도 있음
            # 일단 response에 담아둠
            state["response"] = str(response.content).strip()
            logger.info("도구 없이 직접 응답 생성")

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
    logger.info("--- [NODE] Grade Documents Start ---")
    question = state.get("previous_question") or state.get("question", "")
    
    # State에서 문서 가져오기 (Agent Node에서 이미 수집됨)
    retrieved_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", []) or []

    if not retrieved_docs and not qna_docs:
        logger.warning(f"평가할 문서가 없습니다 (RAG 및 QnA 모두 없음).")
        state["_doc_relevance_score"] = 0.0
        state["_doc_relevance_passed"] = False
        return state
    
    if not evaluation_chat_model:
        logger.warning("평가 모델이 없어 기본값으로 처리합니다.")
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
        return state
    
    try:
        # 평가 대상 구성
        # RAG 문서는 최대 5개, QnA 문서는 최대 3개로 제한 (토큰 고려)
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
        
        # [Async] invoke -> ainvoke
        response = await evaluation_chat_model.ainvoke(messages)
        response_text = response.content.strip()
        
        # [수정] 공통 함수 사용
        evaluation_result = _parse_json_from_response(response_text)
        
        logger.info(f"관련성 평가 결과: {evaluation_result}")
        score = float(evaluation_result.get("score", 0.5))
        reason = evaluation_result.get("reason", "")
        
        # 관련성 있는 문서 인덱스 추출 (1-based index)
        relevant_indices = evaluation_result.get("relevant_indices", [])
        logger.info(f"관련 문서 인덱스: {relevant_indices}")

        state["_doc_relevance_score"] = max(0.0, min(1.0, score))
        state["_doc_relevance_passed"] = score >= 0.6
        logger.info(f"문서 관련성 평가: 점수={score:.2f}, 통과={state['_doc_relevance_passed']}")
        
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
        
        # 필터링된 문서가 없으면 점수가 높아도 실패 처리
        if not filtered_rag and not filtered_qna:
            logger.warning("관련 문서가 하나도 없어 평가를 실패 처리합니다.")
            state["_doc_relevance_passed"] = False
            state["_doc_relevance_score"] = 0.0
        
        # (출처 업데이트 로직 제거 - Service에서 처리)
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
    
    return state


async def analyze_missing_info_node(state: AgentState) -> AgentState:
    """
    Analyze Missing Info Node
    문서가 불충분할 때 사용자에게 필요한 정보를 되묻는 응답 생성
    """
    logger.info("--- [NODE] Analyze Missing Info Start ---")
    question = state.get("previous_question") or state.get("question", "")
    baby_info = state.get("baby_info", {})
    
    if not agent_chat_model:
        state["response"] = "죄송합니다. 현재 정보를 찾을 수 없어 답변이 어렵습니다."
        return state
        
    baby_context = get_baby_context_string(baby_info)
    
    prompt = ANALYZE_MISSING_INFO_PROMPT_TEMPLATE.format(
        question=question,
        baby_context=baby_context
    )
    
    try:
        # [Async] invoke -> ainvoke
        response = await agent_chat_model.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # JSON 파싱
        result = _parse_json_from_response(response_text)
        
        # 1. 사용자 응답 메시지 추출
        generated_response = result.get("response", "죄송합니다. 정확한 답변을 위해 추가 정보가 필요합니다.")
        
        # 2. 누락 정보 리스트 추출
        missing_info_list = result.get("missing_info", [])
        
        logger.info(f"부족한 정보 요청 응답 생성 완료")
        logger.info(f"누락 정보 목록: {missing_info_list}")
        
        # 응답 설정 (순수 텍스트만)
        state["response"] = generated_response
        state["messages"] = [AIMessage(content=generated_response)]
        
        # missing_info 필드에 딕셔너리로 저장 (원래 질문 보존)
        state["_missing_info"] = {
            "missing_info": missing_info_list,
            "pending_question": question
        }
        
    except Exception as e:
        logger.error(f"부족한 정보 분석 실패: {e}")
        state["response"] = "죄송합니다. 요청하신 내용을 파악하기 위해 더 자세한 정보가 필요합니다."
        state["_missing_info"] = None # 오류 발생 시 초기화
        
    return state


async def generate_node(state: AgentState) -> AgentState:
    """
    Generate Node (Self-RAG)
    검색된 문서를 바탕으로 최종 답변 생성 (Strategy B 제거 -> 통합 로직)
    """
    logger.info("--- [NODE] Generate Answer Start ---")
    previous_question = state.get("previous_question") or state.get("question", "")
    baby_info = state.get("baby_info", {})
    
    # evaluate_node에서 필터링된 문서들 가져오기
    retrieved_docs = state.get("_retrieved_docs", [])
    qna_docs = state.get("_qna_docs", [])

    if not agent_chat_model:
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다."
        return state
    
    try:
        baby_context = get_baby_context_string(baby_info)
        
        # --- Prompt Selection Logic ---
        prompt = ""
        mode_log = "Normal"
        
        # [수정] 응급 상황 처리 (최우선)
        if state.get("is_emergency"):
            mode_log = "Emergency"
            logger.info("🚨 Emergency Mode: 응급 프롬프트 적용")
            
            docs_context = get_docs_context_string(retrieved_docs)
            formatted_qna = format_qna_docs(qna_docs) if qna_docs else ""
            
            # 컨텍스트 합치기
            full_context = ""
            if formatted_qna:
                full_context += f"[QnA 정보]\n{formatted_qna}\n\n"
            if docs_context:
                full_context += f"[검색된 문서]\n{docs_context}"
                
            prompt = EMERGENCY_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                full_context=full_context,
                previous_question=previous_question
            )
        else:
            # 통합된 일반 생성 로직 (Green/Yellow/Red 구분 없음)
            logger.info("📝 Standard Generation Mode")
            
            docs_context = ""
            
            # QnA 내용 추가
            if qna_docs:
                formatted_qna = format_qna_docs(qna_docs)
                docs_context += f"[QnA 정보]\n{formatted_qna}\n\n"
                
            # RAG 문서 내용 추가
            if retrieved_docs:
                rag_context = get_docs_context_string(retrieved_docs)
                docs_context += f"{rag_context}"
                
            if not docs_context:
                docs_context = "관련된 참조 문서가 없습니다. 당신의 전문 지식으로 답변해주세요."
            
            prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                docs_context=docs_context
            )
        
        # [로깅] 최종 사용된 출처 정보 출력
        log_sources = []
        
        # QnA 소스 로깅
        for doc in qna_docs:
            filename = getattr(doc, 'source', 'unknown')
            q_text = getattr(doc, 'question', '')
            if len(q_text) > 15:
                q_text = q_text[:15] + "..."
            log_sources.append(f"QnA '{q_text}': {filename}")
            
        # Doc 소스 로깅
        for doc in retrieved_docs:
             filename = getattr(doc, 'filename', 'unknown')
             log_sources.append(f"Doc:{filename}")

        if log_sources:
            logger.info(f"📚 최종 사용된 출처 ({len(log_sources)}개): {', '.join(log_sources)}")
        else:
            logger.info("📚 사용된 출처 없음")

        # 답변 생성
        # [Async] invoke -> ainvoke
        response = await agent_chat_model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=previous_question)
        ])
        
        generated_response = response.content.strip()
        state["response"] = generated_response
        state["is_emergency"] = False
        
        # [추가] 답변이 생성되었으므로, 부족한 정보 요청 상태 초기화
        state["_missing_info"] = None 
        
        # 메시지에 추가
        state["messages"] = [response]
        
        logger.info(f"답변 생성 완료 (모드: {mode_log})")
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["is_emergency"] = False
        # 실패 시에도 상태를 초기화할지 여부는 선택사항이나, 안전하게 초기화
        state["_missing_info"] = None
    
    return state
