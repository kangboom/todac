"""
PyMuPDF (fitz) + pymupdf4llm을 사용하여 PDF를 Markdown으로 변환하는 파서
"""
from typing import List, Dict, Any
from io import BytesIO
from app.services.parsers.base import BaseParser
from app.dto.knowledge import ParsedDocument
import logging

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """PyMuPDF (fitz) + pymupdf4llm을 사용하여 PDF를 Markdown으로 변환하는 파서"""
    
    def parse(self, content: bytes, filename: str = None) -> List[ParsedDocument]:
        """
        PyMuPDF를 사용하여 PDF를 Markdown으로 변환
        
        Args:
            content: PDF 파일 바이너리
            filename: 파일명
        
        Returns:
            ParsedDocument 리스트
        """
        try:
            import fitz  # PyMuPDF
            from pymupdf4llm import to_markdown
            
            logger.info(f"PyMuPDF를 사용하여 파싱을 시도합니다: 파일={filename}")
            
            # PDF 열기
            pdf_document = fitz.open(stream=BytesIO(content), filetype="pdf")
            
            try:
                # pymupdf4llm을 사용하여 Markdown으로 변환
                markdown_text = to_markdown(pdf_document)
                
                if not markdown_text or not markdown_text.strip():
                    logger.warning(f"PyMuPDF가 빈 결과를 반환했습니다: 파일={filename}")
                    raise ValueError(
                        "PDF에서 텍스트를 추출할 수 없습니다. "
                        "PDF가 텍스트 레이어를 포함하지 않거나 이미지로만 구성되어 있을 수 있습니다."
                    )
                
                page_count = len(pdf_document)
                logger.info(f"PyMuPDF 파싱 성공: 파일={filename}, 페이지 수={page_count}, 텍스트 길이={len(markdown_text)}")
                
                # 단일 문서로 반환 (청킹은 별도로 처리)
                return [ParsedDocument(
                    text=markdown_text.strip(),
                    metadata={
                        "filename": filename or "unknown.pdf",
                        "format": "markdown",
                        "parser": "pymupdf"
                    }
                )]
                
            finally:
                pdf_document.close()
                
        except ImportError as e:
            error_msg = str(e)
            logger.error(
                f"PyMuPDF 또는 pymupdf4llm 패키지가 설치되지 않았습니다: {error_msg}. "
                f"pip install pymupdf pymupdf4llm로 설치하세요."
            )
            raise ImportError(
                f"PyMuPDF 또는 pymupdf4llm 패키지가 설치되지 않았습니다: {error_msg}"
            )
        except Exception as e:
            logger.error(
                f"PyMuPDF 파싱 실패: {str(e)}, 파일={filename}. "
                f"에러 타입: {type(e).__name__}"
            )
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            raise
    
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

