"""
Docling을 사용하여 PDF를 Markdown으로 변환하는 파서
"""
from typing import List, Dict, Any
import tempfile
import os
from app.services.parsers.base import BaseParser
import logging

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Docling을 사용하여 PDF를 Markdown으로 변환하는 파서"""
    
    def parse(self, content: bytes, filename: str = None) -> List[Dict[str, Any]]:
        """
        Docling을 사용하여 PDF를 Markdown으로 변환
        
        Args:
            content: PDF 파일 바이너리
            filename: 파일명
        
        Returns:
            청크 리스트 [{"text": str (markdown), "metadata": dict}]
        """
        try:
            from docling.document_converter import DocumentConverter
            
            logger.info(f"Docling을 사용하여 파싱을 시도합니다: 파일={filename}")
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # DocumentConverter 초기화
                converter = DocumentConverter()
                
                # PDF 변환 (파일 경로 사용)
                result = converter.convert(tmp_path)
                
                # Markdown 텍스트 추출
                markdown_text = result.document.export_to_markdown()
                
                if not markdown_text or not markdown_text.strip():
                    logger.warning(f"Docling이 빈 결과를 반환했습니다: 파일={filename}")
                    raise ValueError(
                        "PDF에서 텍스트를 추출할 수 없습니다. "
                        "PDF가 텍스트 레이어를 포함하지 않거나 이미지로만 구성되어 있을 수 있습니다."
                    )
                
                logger.info(f"Docling 파싱 성공: 파일={filename}, 텍스트 길이={len(markdown_text)}")
                
                # 단일 문서로 반환 (청킹은 별도로 처리)
                return [{
                    "text": markdown_text.strip(),
                    "metadata": {
                        "filename": filename or "unknown.pdf",
                        "format": "markdown",
                        "parser": "docling"
                    }
                }]
                    
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
        except ImportError as e:
            error_msg = str(e)
            logger.error(
                f"docling 패키지가 설치되지 않았습니다: {error_msg}. "
                f"pip install docling로 설치하세요."
            )
            raise ImportError(
                f"docling 패키지가 설치되지 않았습니다: {error_msg}"
            )
        except Exception as e:
            logger.error(
                f"Docling 파싱 실패: {str(e)}, 파일={filename}. "
                f"에러 타입: {type(e).__name__}"
            )
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            raise
    
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

