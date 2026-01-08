"""
Docling을 사용하여 PDF를 Markdown으로 변환하는 파서
"""
from typing import List, Dict, Any
import tempfile
import os

from sqlalchemy.sql import True_
from app.services.parsers.base import BaseParser
import logging
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption, DocumentConverter

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Docling을 사용하여 PDF를 Markdown으로 변환하는 파서"""
    
    def __init__(self):
        try:
            # 1. 파이프라인 옵션 구성
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False           # OCR 비활성화
            pipeline_options.do_table_structure = True # 테이블 구조 인식 켜기
            
            # 2. Converter 생성 시 포맷 옵션으로 전달
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling DocumentConverter가 초기화되었습니다 (OCR: Disabled, Table: True).")
        except Exception as e:
            logger.error(f"Docling 초기화 실패: {str(e)}")
            self.converter = None

    def parse(self, content: bytes, filename: str = None) -> List[Dict[str, Any]]:
        """
        Docling을 사용하여 PDF를 Markdown으로 변환
        
        Args:
            content: PDF 파일 바이너리
            filename: 파일명
        
        Returns:
            청크 리스트 [{"text": str (markdown), "metadata": dict}]
        """
        if not self.converter:
            raise ImportError(
                "docling 패키지가 설치되지 않았거나 초기화에 실패했습니다. "
                "pip install docling로 설치하세요."
            )

        try:
            logger.info(f"Docling을 사용하여 파싱을 시도합니다: 파일={filename}")
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # PDF 변환 (초기화된 converter 재사용)
                result = self.converter.convert(tmp_path)
                
                # 문서 객체 가져오기
                doc = result.document
                
                # 헤더와 푸터 라벨 정의
                from docling_core.types.doc import DocItemLabel
                
                # DoclingDocument를 순회하며 HEADER/FOOTER 라벨을 가진 아이템의 내용을 비움
                for item, _ in doc.iterate_items():
                    # 모든 아이템의 라벨과 텍스트 로깅 (디버깅용)
                    # 텍스트가 있는 경우에만 로깅
                    if hasattr(item, "text"):
                        log_text = item.text.strip()[:50] + "..." if len(item.text.strip()) > 50 else item.text.strip()
                        logger.info(f"Docling Item: Label={item.label}, Text='{log_text}'")
                    else:
                        logger.info(f"Docling Item: Label={item.label}, No text attribute")

                    if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                        # 텍스트를 비워서 export 시 포함되지 않게 함
                        if hasattr(item, "text"):
                            logger.info(f"헤더/푸터 제거됨 ({item.label}): {item.text.strip()}")
                            item.text = ""
                            
                # 수정된 문서 객체로 Markdown 변환
                markdown_text = doc.export_to_markdown()
                
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

