"""
노드 함수 (Self-RAG 구조)
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from app.agent.state import AgentState
from app.agent.prompts import (
    SYSTEM_PROMPT,  # Agent Node용
    DOC_RELEVANCE_PROMPT_TEMPLATE, 
    REWRITE_QUERY_PROMPT_TEMPLATE,
    HALLUCINATION_CHECK_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    AGENT_NODE_PROMPT_TEMPLATE,
    get_baby_context_string,
    get_docs_context_string,
    PERSONA_PROMPT, # 공통 페르소나
    QNA_GREEN_PROMPT_TEMPLATE,
    QNA_YELLOW_PROMPT_TEMPLATE,
    EMERGENCY_PROMPT_TEMPLATE # [변경] 응급 상황 프롬프트 템플릿
)
from app.agent.tools import milvus_knowledge_search, report_emergency
from app.services.qna_service import search_qna, format_qna_docs
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
    max_tokens=200
) if settings.OPENAI_API_KEY else None


def retrieve_qna_node(state: AgentState) -> AgentState:
    """
    공식 QnA 검색 노드
    가장 먼저 실행되어 QnA DB를 검색하고 점수를 확인합니다.
    """
    logger.info("--- [NODE] QnA Retrieval Start ---")
    question = state.get("original_question") or state.get("question", "")
    
    # 원본 질문이 없으면 저장
    if not state.get("original_question"):
        state["original_question"] = question

    # QnA 검색 실행 (동기 호출)
    qna_results = search_qna(question)
    
    # 최고 점수 계산
    max_score = 0.0
    if qna_results:
        # DTO 객체이므로 .score 속성 접근
        max_score = max([doc.score for doc in qna_results])
        
    logger.info(f"QnA Search Result: Score={max_score:.2f}, Count={len(qna_results)}")
    
    # State 업데이트
    state["qna_docs"] = qna_results
    state["qna_score"] = max_score
    
    return state


def agent_node(state: AgentState) -> AgentState:
    """
    핵심 에이전트 노드 (Self-RAG)
    - 질문 분석 및 tool 호출 결정
    - Tool 호출이 필요하면 tool 호출, 없으면 직접 답변
    """
    logger.info("--- [NODE] Agent Analysis Start ---")

    question = state.get("question", "")
    messages = state.get("messages", [])
    baby_info = state.get("baby_info", {})
    
    if not agent_chat_model:
        logger.error("OpenAI 클라이언트가 없어 에이전트를 실행할 수 없습니다.")
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."
        return state
    
    try:
        # [수정] bind_tools 사용하여 툴 바인딩 (표준 Tool Calling 방식)
        tools = [
            milvus_knowledge_search,  # RAG 검색 tool
            report_emergency,         # 응급 상태 보고 tool
        ]
        model_with_tools = agent_chat_model.bind_tools(tools)
        
        # 시스템 프롬프트 생성 (아기 정보 포함)
        baby_context = get_baby_context_string(baby_info)
        
        system_prompt = AGENT_NODE_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            baby_context=baby_context
        )
        
        # 시스템 메시지 추가
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
        
        # Agent 실행
        response = model_with_tools.invoke(messages_with_system)
        
        # [로직] 툴 호출 확인하여 is_emergency 플래그 설정
        state["is_emergency"] = False # 초기화
        
        # response가 tool_calls 속성을 가지고 있는지 확인 (Pydantic v1/v2 호환성)
        has_tool_calls = False
        if hasattr(response, 'tool_calls') and response.tool_calls:
            has_tool_calls = True
        elif isinstance(response, dict) and response.get('tool_calls'):
            has_tool_calls = True
            
        if has_tool_calls:
            tool_calls = getattr(response, 'tool_calls', []) or response.get('tool_calls', [])
            
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                logger.info(f"🛠️ Tool Call 감지: {tool_name} (Args: {tool_call.get('args')})")
                
                # 응급 툴이 호출되면 플래그 True 설정
                if tool_name == 'report_emergency':
                    logger.info(f"🚨 응급 툴 호출 감지 -> 응급 모드 활성화")
                    state["is_emergency"] = True
            
            tool_calls_count = len(tool_calls)
            logger.info(f"Tool 호출 결정: {tool_calls_count}개 tool 호출")
            
        else:
            # Tool 호출이 없으면 직접 답변 (AIMessage content 사용)
            state["response"] = str(response.content).strip()
            logger.info("도구 없이 직접 응답 생성")

        # 응답을 메시지에 추가
        state["messages"] = [response]
        
        # Tool 호출이 없으면 직접 답변 (중복 로직 제거 및 정리)
        # should_continue 노드에서 tool_calls 유무로 판단함
        
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        state["is_emergency"] = False
    
    return state


def grade_documents_node(state: AgentState) -> AgentState:
    """
    Grade Documents Node (Self-RAG)
    검색된 문서의 질문 관련성을 평가
    """
    logger.info("--- [NODE] Grade Documents Start ---")
    question = state.get("original_question") or state.get("question", "")
    messages = state.get("messages", [])
    
    # 먼저 state에서 확인
    retrieved_docs = state.get("retrieved_docs", [])
    
    # State에 없으면 ToolMessage에서 추출
    if not retrieved_docs:
        logger.info(f"ToolMessage 추출 시작: messages 개수={len(messages)}")
        for idx, msg in enumerate(reversed(messages)):
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                if isinstance(tool_result, list) and tool_result:
                    retrieved_docs = tool_result
                    logger.info(f"검색 결과 추출 성공: {len(retrieved_docs)}개 문서")
                    state["retrieved_docs"] = retrieved_docs
                    
                    # RAG 소스 정보 저장
                    rag_sources = [
                        {
                            "doc_id": str(doc.get("doc_id", "")),
                            "chunk_index": doc.get("chunk_index", ""),
                            "score": doc.get("score", 0.0),
                            "filename": doc.get("filename", ""),
                            "category": doc.get("category", "")
                        }
                        for doc in retrieved_docs
                    ]
                    state["rag_sources"] = rag_sources
                    break
                # JSON 문자열로 직렬화된 경우
                elif isinstance(tool_result, str):
                     try:
                        import json
                        parsed_result = json.loads(tool_result)
                        if isinstance(parsed_result, list) and parsed_result:
                            retrieved_docs = parsed_result
                            logger.info(f"JSON 파싱 성공: {len(retrieved_docs)}개 문서")
                            state["retrieved_docs"] = retrieved_docs
                            
                            rag_sources = [
                                {
                                    "doc_id": str(doc.get("doc_id", "")),
                                    "chunk_index": doc.get("chunk_index", ""),
                                    "score": doc.get("score", 0.0),
                                    "filename": doc.get("filename", ""),
                                    "category": doc.get("category", "")
                                }
                                for doc in retrieved_docs
                            ]
                            state["rag_sources"] = rag_sources
                            break
                     except (json.JSONDecodeError, TypeError):
                        pass

    if not retrieved_docs:
        logger.warning(f"평가할 문서가 없습니다.")
        state["_doc_relevance_score"] = 0.0
        state["_doc_relevance_passed"] = False
        return state
    
    if not evaluation_chat_model:
        logger.warning("평가 모델이 없어 기본값으로 처리합니다.")
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
        return state
    
    try:
        # 상위 3개 문서만 평가
        docs_to_evaluate = retrieved_docs[:3]
        
        docs_summary = ""
        for i, doc in enumerate(docs_to_evaluate, 1):
            content = doc.get('content', '')[:300]
            docs_summary += f"\n문서 {i}:\n{content}...\n"
        
        evaluation_prompt = DOC_RELEVANCE_PROMPT_TEMPLATE.format(
            question=question,
            docs_summary=docs_summary
        )
        
        messages = [
            SystemMessage(content="당신은 문서 관련성을 평가하는 전문가입니다. 객관적이고 정확하게 평가하세요."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        response = evaluation_chat_model.invoke(messages)
        response_text = response.content.strip()
        
        # JSON 파싱 및 점수 추출
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation_result = json.loads(response_text)
            logger.info(f"관련성 평가 결과: {evaluation_result}")
            score = float(evaluation_result.get("score", 0.5))
            reason = evaluation_result.get("reason", "")
            
            # [추가] 관련성 있는 문서 인덱스 추출 (1-based index)
            relevant_indices = evaluation_result.get("relevant_indices", [])
            
            logger.info(f"관련 문서 인덱스: {relevant_indices}")

            state["_doc_relevance_score"] = max(0.0, min(1.0, score))
            state["_doc_relevance_passed"] = score >= 0.6
            logger.info(f"문서 관련성 평가: 점수={score:.2f}, 통과={state['_doc_relevance_passed']}")
            
            # [수정] 관련성 있는 문서만 필터링하여 저장 (Pass 여부와 상관없이 항상 적용)
            # relevant_indices가 비어있으면 retrieved_docs도 빈 리스트가 됨 -> 화면에 엉뚱한 문서 표시 방지
            filtered_docs = []
            if relevant_indices:
                # indices는 1부터 시작하므로 -1 해줌
                for idx in relevant_indices:
                    if 1 <= idx <= len(docs_to_evaluate):
                        filtered_docs.append(docs_to_evaluate[idx-1])
            
            # 필터링 결과 적용
            logger.info(f"관련성 필터링: {len(retrieved_docs)}개 -> {len(filtered_docs)}개")
            state["retrieved_docs"] = filtered_docs
            
            # [추가] 필터링된 문서가 없으면 점수가 높아도 실패 처리
            if not filtered_docs:
                logger.warning("관련 문서가 없어 평가를 실패 처리합니다.")
                state["_doc_relevance_passed"] = False
                state["_doc_relevance_score"] = 0.0
            
            # RAG 소스 정보 재구성
            if filtered_docs:
                rag_sources = [
                    {
                        "doc_id": str(doc.get("doc_id", "")),
                        "chunk_index": doc.get("chunk_index", ""),
                        "score": doc.get("score", 0.0),
                        "filename": doc.get("filename", ""),
                        "category": doc.get("category", "")
                    }
                    for doc in filtered_docs
                ]
                state["rag_sources"] = rag_sources
            else:
                state["rag_sources"] = []

        except Exception as e:
            logger.error(f"JSON 파싱 실패: {str(e)}")
            state["_doc_relevance_score"] = 0.5
            state["_doc_relevance_passed"] = True
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True
    
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrite Query Node"""
    logger.info("--- [NODE] Rewrite Query Start ---")
    original_question = state.get("original_question") or state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    # [추가] RAG 검색 시도 횟수 증가
    attempts = state.get("rag_retrieval_attempts", 0) + 1
    state["rag_retrieval_attempts"] = attempts
    logger.info(f"RAG 검색 재시도 횟수: {attempts}")

    state["retrieved_docs"] = [] # 재검색을 위해 초기화
    
    if not agent_chat_model:
        return state
    
    try:
        docs_summary = ""
        if retrieved_docs:
            docs_summary = "\n이전 검색 결과 (관련성이 낮았음):\n"
            for i, doc in enumerate(retrieved_docs[:2], 1):
                content = doc.get('content', '')[:150]
                docs_summary += f"{i}. {content}...\n"
        
        rewrite_prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(
            original_question=original_question,
            docs_summary=docs_summary
        )
        
        messages = [
            SystemMessage(content="당신은 검색 쿼리 최적화 전문가입니다."),
            HumanMessage(content=rewrite_prompt)
        ]
        
        response = agent_chat_model.invoke(messages)
        rewritten_query = response.content.strip()
        
        if not state.get("original_question"):
            state["original_question"] = original_question
        
        state["question"] = rewritten_query
        logger.info(f"질문 재구성: '{original_question}' → '{rewritten_query}'")
        
        # 재구성된 질문을 HumanMessage로 추가하여 Agent가 다시 검색하도록 유도
        new_message = HumanMessage(
            content=f"이전 검색 결과가 충분하지 않습니다. 질문을 다음과 같이 구체화하여 다시 검색해주세요: '{rewritten_query}'"
        )
        # add_messages 리듀서가 동작하도록 리스트로 감싸서 반환
        state["messages"] = [new_message]
        
    except Exception as e:
        logger.error(f"질문 재구성 실패: {str(e)}")
    
    return state


def generate_node(state: AgentState) -> AgentState:
    """
    Generate Node (Self-RAG + Strategy B)
    검색된 문서를 바탕으로 최종 답변 생성
    """
    logger.info("--- [NODE] Generate Answer Start ---")
    original_question = state.get("original_question") or state.get("question", "")
    baby_info = state.get("baby_info", {})
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    
    # Strategy B State
    qna_score = state.get("qna_score", 0.0)
    qna_docs = state.get("qna_docs", [])
    
    attempts = state.get("_generation_attempts", 0) + 1
    state["_generation_attempts"] = attempts
    
    if not agent_chat_model:
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다."
        return state
    
    try:
        baby_context = get_baby_context_string(baby_info)
        
        # --- Prompt Selection Logic ---
        prompt = ""
        mode_log = "Red"
        
        # 로깅용 변수
        log_context = ""

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
                original_question=original_question
            )
        elif qna_score >= 0.9 and qna_docs:
            mode_log = "Green"
            logger.info("🟢 Green Mode: QnA Only Generation")
            
            formatted_qna = format_qna_docs(qna_docs)
            log_context = formatted_qna
            
            prompt = QNA_GREEN_PROMPT_TEMPLATE.format(
                question=original_question,
                qna_context=formatted_qna
            )
            
        elif qna_score >= 0.7 and qna_docs:
            mode_log = "Yellow"
            logger.info("🟡 Yellow Mode: Hybrid Generation")
            
            docs_context = get_docs_context_string(retrieved_docs)
            formatted_qna = format_qna_docs(qna_docs)
            log_context = f"QnA:\n{formatted_qna}\n\nDocs:\n{docs_context}"
            
            prompt = QNA_YELLOW_PROMPT_TEMPLATE.format(
                baby_context=baby_context,
                question=original_question,
                qna_context=formatted_qna,
                context=docs_context
            )
            
        else:
            # Red or Normal Mode
            logger.info("🔴 Red/Normal Mode: Standard RAG Generation")
            docs_context = get_docs_context_string(retrieved_docs)
            log_context = docs_context
            
            # RAG 소스 정보 저장 (일반 검색만 해당)
            if retrieved_docs:
                rag_sources = [
                    {
                        "doc_id": str(doc.get("doc_id", "")),
                        "chunk_index": doc.get("chunk_index", ""),
                        "score": doc.get("score", 0.0),
                        "filename": doc.get("filename", ""),
                        "category": doc.get("category", "")
                    }
                    for doc in retrieved_docs
                ]
                state["rag_sources"] = rag_sources
            
            prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
                system_prompt=PERSONA_PROMPT,
                baby_context=baby_context,
                docs_context=docs_context
            )
        
        # 답변 생성
        response = agent_chat_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=original_question)
        ])
        
        generated_response = response.content.strip()
        state["response"] = generated_response
        state["is_emergency"] = False
        
        # 메시지에 추가
        state["messages"] = [response]
        
        logger.info(f"답변 생성 완료 (모드: {mode_log}, 시도: {attempts})")
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}", exc_info=True)
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["is_emergency"] = False
    
    return state


def grade_hallucination_node(state: AgentState) -> AgentState:
    """
    Grade Hallucination Node
    """
    logger.info("--- [NODE] Grade Hallucination Start ---")
    question = state.get("original_question") or state.get("question", "")
    response = state.get("response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    qna_docs = state.get("qna_docs", [])
    qna_score = state.get("qna_score", 0.0)
    
    if not response:
        state["_hallucination_score"] = 0.0
        state["_hallucination_passed"] = False
        return state
    
    if not evaluation_chat_model:
        state["_hallucination_score"] = 0.8
        state["_hallucination_passed"] = True
        return state
    
    try:
        # 검증 대상 문서 선택
        context_docs = []
        mode_log = "Red"
        
        if qna_score >= 0.9 and qna_docs:
            mode_log = "Green"
            # QnADoc는 Pydantic 모델
            docs_summary = "\n참조 문서 (QnA):\n"
            for i, doc in enumerate(qna_docs[:3], 1):
                docs_summary += f"{i}. Q: {doc.question}\nA: {doc.answer}\n"
        elif qna_score >= 0.7:
            mode_log = "Yellow"
            docs_summary = "\n참조 문서 (QnA + General):\n"
            if qna_docs:
                for i, doc in enumerate(qna_docs[:2], 1):
                    docs_summary += f"QnA {i}: {doc.answer[:100]}...\n"
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs[:2], 1):
                    content = doc.get('content', '')
                    docs_summary += f"Doc {i}: {content[:100]}...\n"
        else:
            mode_log = "Red"
            docs_summary = "\n참조 문서:\n"
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    content = doc.get('content', '')
                    docs_summary += f"{i}. {content[:200]}...\n"
            else:
                docs_summary = "\n참조 문서가 없습니다.\n"
        
        evaluation_prompt = HALLUCINATION_CHECK_PROMPT_TEMPLATE.format(
            question=question,
            docs_summary=docs_summary,
            response=response
        )
        
        messages = [
            SystemMessage(content="당신은 답변의 정확성과 환각을 평가하는 전문가입니다."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        eval_response = evaluation_chat_model.invoke(messages)
        response_text = eval_response.content.strip()
        
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation_result = json.loads(response_text)
            score = float(evaluation_result.get("score", 0.5))
            has_hallucination = evaluation_result.get("has_hallucination", False)
            
            state["_hallucination_score"] = max(0.0, min(1.0, score))
            state["_hallucination_passed"] = score >= 0.7 and not has_hallucination
            
            logger.info(f"환각 평가 ({mode_log}): 통과={state['_hallucination_passed']}")
            
        except Exception:
            state["_hallucination_score"] = 0.7
            state["_hallucination_passed"] = True
        
    except Exception as e:
        logger.error(f"환각 평가 실패: {str(e)}")
        state["_hallucination_score"] = 0.7
        state["_hallucination_passed"] = True
    
    return state
