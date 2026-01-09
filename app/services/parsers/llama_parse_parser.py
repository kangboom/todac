"""
LlamaParse를 사용하여 PDF를 Markdown으로 변환하는 파서
"""
from typing import List, Dict, Any
import tempfile
import os
from app.services.parsers.base import BaseParser
from app.dto.knowledge import ParsedDocument
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """LlamaParse를 사용하여 PDF를 Markdown으로 변환하는 파서"""
    
    def parse(self, content: bytes, filename: str = None) -> List[ParsedDocument]:
        """
        LlamaParse를 사용하여 PDF를 Markdown으로 변환
        
        Args:
            content: PDF 파일 바이너리
            filename: 파일명
        
        Returns:
            ParsedDocument 리스트
        """
        if not settings.LLAMAPARSE_API_KEY:
            raise ValueError(
                "LlamaParse API 키가 설정되지 않았습니다. "
                "환경 변수 LLAMAPARSE_API_KEY를 설정해주세요."
            )
        
        try:
            from llama_parse import LlamaParse
            
            logger.info(f"LlamaParse를 사용하여 파싱을 시도합니다: 파일={filename}")
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # LlamaParse 초기화
                parser = LlamaParse(
                    api_key=settings.LLAMAPARSE_API_KEY,
                    result_type="markdown",  # Markdown 형식으로 반환
                    verbose=True,
                    num_workers=4  # 병렬 처리 워커 수
                )
                
                # PDF 파싱
                documents = parser.load_data(tmp_path)
                
                # Markdown 텍스트 추출
                markdown_text = None
                if documents:
                    # documents가 리스트인 경우 (여러 페이지)
                    if isinstance(documents, list) and len(documents) > 0:
                        # 모든 문서의 텍스트를 합침
                        markdown_parts = []
                        for doc in documents:
                            doc_text = (
                                getattr(doc, 'text', None) or 
                                getattr(doc, 'page_content', None) or 
                                getattr(doc, 'content', None) or 
                                str(doc)
                            )
                            if doc_text and doc_text.strip():
                                markdown_parts.append(doc_text.strip())
                        
                        # 모든 페이지를 개행으로 연결
                        markdown_text = '\n\n'.join(markdown_parts) if markdown_parts else None
                        
                        if markdown_text:
                            logger.info(f"LlamaParse 파싱 성공: 파일={filename}, 페이지 수={len(documents)}, 텍스트 길이={len(markdown_text)}")
                    # 단일 문자열인 경우
                    elif isinstance(documents, str):
                        markdown_text = documents
                    # 단일 Document 객체인 경우
                    else:
                        markdown_text = (
                            getattr(documents, 'text', None) or 
                            getattr(documents, 'page_content', None) or 
                            getattr(documents, 'content', None) or 
                            str(documents)
                        )
                    
                    if markdown_text and markdown_text.strip():
                        logger.info(f"LlamaParse 파싱 성공: 파일={filename}, 텍스트 길이={len(markdown_text)}")
                
                if not markdown_text or not markdown_text.strip():
                    logger.warning(f"LlamaParse가 빈 결과를 반환했습니다: 파일={filename}")
                    raise ValueError(
                        "PDF에서 텍스트를 추출할 수 없습니다. "
                        "PDF가 텍스트 레이어를 포함하지 않거나 이미지로만 구성되어 있을 수 있습니다."
                    )
                
                # 단일 문서로 반환 (청킹은 별도로 처리)
                return [ParsedDocument(
                    text=markdown_text.strip(),
                    metadata={
                        "filename": filename or "unknown.pdf",
                        "format": "markdown",
                        "parser": "llamaparse"
                    }
                )]
                    
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except ImportError as e:
            error_msg = str(e)
            logger.error(
                f"llama-parse 패키지가 설치되지 않았습니다: {error_msg}. "
                f"pip install llama-parse로 설치하세요."
            )
            raise ImportError(
                f"llama-parse 패키지가 설치되지 않았습니다: {error_msg}"
            )
        except Exception as e:
            logger.error(
                f"LlamaParse 파싱 실패: {str(e)}, 파일={filename}. "
                f"에러 타입: {type(e).__name__}"
            )
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            raise
    
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

