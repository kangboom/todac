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
    PERSONA_PROMPT # 공통 페르소나
)
from app.agent.tools import milvus_knowledge_search, emergency_protocol_handler
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


def agent_node(state: AgentState) -> AgentState:
    """
    핵심 에이전트 노드 (Self-RAG)
    - 질문 분석 및 tool 호출 결정
    - Tool 호출이 필요하면 tool 호출, 없으면 직접 답변
    """
    question = state.get("question", "")
    messages = state.get("messages", [])
    baby_info = state.get("baby_info", {})
    
    # 원본 질문이 없으면 현재 질문을 원본으로 저장
    if not state.get("original_question"):
        state["original_question"] = question
    
    if not agent_chat_model:
        logger.error("OpenAI 클라이언트가 없어 에이전트를 실행할 수 없습니다.")
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."
        return state
    
    try:
        # 모든 Tool을 바인딩한 모델 생성
        tools = [
            milvus_knowledge_search,  # RAG 검색 tool
            emergency_protocol_handler  # 응급 처리 tool
        ]
        model_with_tools = agent_chat_model.bind_tools(tools)
        
        # ToolMessage가 있는지 확인 (tool 실행 후 재호출인지)
        # agent_node는 라우팅 역할만 하므로, 이전 검색 결과(retrieved_docs)를
        # 프롬프트에 주입하지 않음 (rewrite 루프 시 오염 방지)
        emergency_response_text = None
        
        # ToolMessage에서 응급 응답만 확인 (검색 결과는 무시)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                if isinstance(tool_result, str) and ("응급 상황" in tool_result or "119" in tool_result) and not emergency_response_text:
                    emergency_response_text = tool_result
        
        # 응급 응답 처리
        if emergency_response_text:
            state["response"] = emergency_response_text
            state["is_emergency"] = True
            logger.info("응급 프로토콜 응답 생성 완료")
            return state
        
        # 시스템 프롬프트 생성 (아기 정보 포함)
        baby_context = get_baby_context_string(baby_info)
        
        # 중요: agent_node에서는 docs_context를 주입하지 않음!
        # 답변 생성은 generate_node의 역할이며, agent_node는 도구 호출 여부만 판단함.
        
        # 시스템 프롬프트 생성
        # AGENT_NODE_PROMPT_TEMPLATE에서 docs_context 제거 필요하지만,
        # 템플릿 호환성을 위해 빈 문자열 전달
        system_prompt = AGENT_NODE_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            baby_context=baby_context,
            docs_context="" # 문서를 보지 않고 판단
        )
        
        # 시스템 메시지 추가
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
        
        # Agent 실행 (tool 호출 결정 또는 최종 응답 생성)
        response = model_with_tools.invoke(messages_with_system)
        
        # 응답을 메시지에 추가 (리듀서 사용: 새 메시지만 추가하면 LangGraph가 자동으로 병합)
        state["messages"] = [response]
        
        # Tool 호출이 없으면 직접 답변 (간단한 질문/인사 등)
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            state["response"] = response.content.strip()
            state["is_emergency"] = False
            logger.info("도구 없이 직접 응답 생성 완료")
        else:
            tool_calls_count = len(response.tool_calls)
            logger.info(f"Tool 호출 결정: {tool_calls_count}개 tool 호출")
        
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
    question = state.get("original_question") or state.get("question", "")
    messages = state.get("messages", [])
    
    # ToolNode 실행 후에는 state["retrieved_docs"]에 값이 없음 (ToolMessage로만 존재)
    # 따라서 ToolMessage에서 추출해야 함
    
    # 먼저 state에서 확인 (재평가 등의 경우 이미 있을 수 있음)
    retrieved_docs = state.get("retrieved_docs", [])
    
    if retrieved_docs:
        logger.info(f"State에서 검색 결과 확인: {len(retrieved_docs)}개 문서")
    
    # State에 없으면 ToolMessage에서 추출 (ToolNode 실행 직후)
    if not retrieved_docs:
        logger.info(f"ToolMessage 추출 시작: messages 개수={len(messages)}")
        for idx, msg in enumerate(reversed(messages)):
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                # logger.debug(f"ToolMessage 발견: content type={type(tool_result)}")
                
                # RAG 검색 결과 (리스트)
                if isinstance(tool_result, list) and tool_result:
                    retrieved_docs = tool_result
                    logger.info(f"검색 결과 추출 성공: {len(retrieved_docs)}개 문서")
                    # state에 저장하여 다음 노드(generate 등)에서 사용 가능하게 함
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
                elif isinstance(tool_result, str):
                     # JSON 문자열로 직렬화된 경우
                     try:
                        import json
                        parsed_result = json.loads(tool_result)
                        if isinstance(parsed_result, list) and parsed_result:
                            retrieved_docs = parsed_result
                            logger.info(f"JSON 파싱 성공: {len(retrieved_docs)}개 문서")
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
                     except (json.JSONDecodeError, TypeError):
                        pass
    
    if not retrieved_docs:
        # 모든 messages를 로그로 출력하여 디버깅
        logger.warning(f"평가할 문서가 없습니다. messages 개수={len(messages)}")
        for idx, msg in enumerate(messages[-5:]):  # 최근 5개만
            msg_type = type(msg).__name__
            if isinstance(msg, dict):
                msg_role = msg.get("role", "unknown")
                msg_content_type = type(msg.get("content", "")).__name__
                logger.warning(f"  [{idx}] dict: role={msg_role}, content_type={msg_content_type}")
            else:
                logger.warning(f"  [{idx}] {msg_type}: {str(msg)[:100]}")
        
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
        
        # 문서 요약
        docs_summary = ""
        for i, doc in enumerate(docs_to_evaluate, 1):
            content = doc.get('content', '')[:300]  # 처음 300자만
            docs_summary += f"\n문서 {i}:\n{content}...\n"
        
        # 평가 프롬프트
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
        
        # JSON 파싱
        try:
            # JSON 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation_result = json.loads(response_text)
            score = float(evaluation_result.get("score", 0.5))
            reason = evaluation_result.get("reason", "")
    
            # 점수 정규화 (0.0 ~ 1.0)
            score = max(0.0, min(1.0, score))
            
            state["_doc_relevance_score"] = score
            # 임계값: 0.6 이상이면 통과
            state["_doc_relevance_passed"] = score >= 0.6
            
            logger.info(f"문서 관련성 평가: 점수={score:.2f}, 통과={state['_doc_relevance_passed']}, 이유={reason}")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"평가 결과 파싱 실패: {str(e)}, 응답: {response_text}")
            # 기본값 설정
            state["_doc_relevance_score"] = 0.5
            state["_doc_relevance_passed"] = True  # 파싱 실패 시 통과로 처리
        
    except Exception as e:
        logger.error(f"문서 평가 실패: {str(e)}", exc_info=True)
        state["_doc_relevance_score"] = 0.5
        state["_doc_relevance_passed"] = True  # 에러 시 통과로 처리
    
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    """
    Rewrite Query Node (Self-RAG)
    질문을 재구성하여 더 나은 검색 결과를 얻기 위해
    """
    original_question = state.get("original_question") or state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    # 재검색을 위해 이전 검색 결과 초기화 (중요: agent_node 재진입 시 오염 방지)
    state["retrieved_docs"] = []
    
    if not agent_chat_model:
        logger.warning("재작성 모델이 없어 원본 질문을 유지합니다.")
        return state
    
    try:
        # 이전 검색 결과 요약
        docs_summary = ""
        if retrieved_docs:
            docs_summary = "\n이전 검색 결과 (관련성이 낮았음):\n"
            for i, doc in enumerate(retrieved_docs[:2], 1):
                content = doc.get('content', '')[:150]
                docs_summary += f"{i}. {content}...\n"
        
        # 프롬프트 생성성
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
        
        # 원본 질문 보존
        if not state.get("original_question"):
            state["original_question"] = original_question
        
        # 재구성된 질문으로 업데이트
        state["question"] = rewritten_query
        
        logger.info(f"질문 재구성: '{original_question}' → '{rewritten_query}'")
        
    except Exception as e:
        logger.error(f"질문 재구성 실패: {str(e)}, 원본 질문 유지")
    
    return state


def generate_node(state: AgentState) -> AgentState:
    """
    Generate Node (Self-RAG)
    검색된 문서를 바탕으로 최종 답변 생성
    """
    original_question = state.get("original_question") or state.get("question", "")
    baby_info = state.get("baby_info", {})
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    
    # 생성 시도 횟수 증가
    attempts = state.get("_generation_attempts", 0) + 1
    state["_generation_attempts"] = attempts
    
    if not agent_chat_model:
        logger.error("생성 모델이 없어 답변을 생성할 수 없습니다.")
        state["response"] = "죄송합니다. 현재 답변을 생성할 수 없습니다."
        return state
    
    try:
        # 아기 정보 컨텍스트
        baby_context = get_baby_context_string(baby_info)
        
        # 참조 문서 정보
        docs_context = get_docs_context_string(retrieved_docs)
        
        # RAG 소스 정보 저장
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
        
        # 시스템 프롬프트
        system_prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
            system_prompt=PERSONA_PROMPT,  # 답변 생성 시에는 Tool 가이드라인 제외
            baby_context=baby_context,
            docs_context=docs_context
        )
        
        # LangChain 메시지 형식으로 변환
        langchain_messages = [HumanMessage(content=original_question)]
        
        # 답변 생성
        messages_with_system = [SystemMessage(content=system_prompt)] + langchain_messages
        response = agent_chat_model.invoke(messages_with_system)
        
        generated_response = response.content.strip()
        state["response"] = generated_response
        state["is_emergency"] = False
        
        # 메시지에 추가
        state["messages"] = [response]
        
        logger.info(f"답변 생성 완료 (시도 {attempts})")
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}")
        state["response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["is_emergency"] = False
    
    return state


def grade_hallucination_node(state: AgentState) -> AgentState:
    """
    Grade Hallucination Node (Self-RAG)
    생성된 답변의 환각(정확도)을 평가
    """
    question = state.get("original_question") or state.get("question", "")
    response = state.get("response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    if not response:
        logger.warning("평가할 답변이 없습니다.")
        state["_hallucination_score"] = 0.0
        state["_hallucination_passed"] = False
        return state
    
    if not evaluation_chat_model:
        logger.warning("평가 모델이 없어 기본값으로 처리합니다.")
        state["_hallucination_score"] = 0.8
        state["_hallucination_passed"] = True
        return state
    
    try:
        # 참조 문서 요약
        docs_summary = ""
        if retrieved_docs:
            docs_summary = "\n참조 문서:\n"
            for i, doc in enumerate(retrieved_docs[:3], 1):
                content = doc.get('content', '')[:200]
                docs_summary += f"{i}. {content}...\n"
        else:
            docs_summary = "\n참조 문서가 없습니다.\n"
        
        # 평가 프롬프트
        evaluation_prompt = HALLUCINATION_CHECK_PROMPT_TEMPLATE.format(
            question=question,
            docs_summary=docs_summary,
            response=response
        )
        
        messages = [
            SystemMessage(content="당신은 답변의 정확성과 환각을 평가하는 전문가입니다. 엄격하게 평가하세요."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        eval_response = evaluation_chat_model.invoke(messages)
        response_text = eval_response.content.strip()
        
        # JSON 파싱
        try:
            # JSON 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation_result = json.loads(response_text)
            score = float(evaluation_result.get("score", 0.5))
            reason = evaluation_result.get("reason", "")
            has_hallucination = evaluation_result.get("has_hallucination", False)
            
            # 점수 정규화 (0.0 ~ 1.0)
            score = max(0.0, min(1.0, score))
            
            state["_hallucination_score"] = score
            # 임계값: 0.7 이상이고 환각이 없으면 통과
            state["_hallucination_passed"] = score >= 0.7 and not has_hallucination
            
            logger.info(f"환각 평가: 점수={score:.2f}, 통과={state['_hallucination_passed']}, 환각={has_hallucination}, 이유={reason}")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"평가 결과 파싱 실패: {str(e)}, 응답: {response_text}")
            # 기본값 설정
            state["_hallucination_score"] = 0.7
            state["_hallucination_passed"] = True  # 파싱 실패 시 통과로 처리
        
    except Exception as e:
        logger.error(f"환각 평가 실패: {str(e)}", exc_info=True)
        state["_hallucination_score"] = 0.7
        state["_hallucination_passed"] = True  # 에러 시 통과로 처리
    
    return state
