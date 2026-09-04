"""Docling PDF parser with bounded, sequential page-range conversion."""

import gc
import logging
import os
import tempfile
from typing import List

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel

from app.core.config import settings
from app.dto.knowledge import ParsedDocument
from app.services.parsers.base import BaseParser

logger = logging.getLogger(__name__)


def page_ranges(total_pages: int, batch_size: int):
    """Yield one-based, inclusive page ranges in document order."""
    if total_pages < 1:
        raise ValueError("PDF에 처리할 페이지가 없습니다.")
    if batch_size < 1:
        raise ValueError("Docling 페이지 배치 크기는 1 이상이어야 합니다.")
    for start_page in range(1, total_pages + 1, batch_size):
        yield start_page, min(start_page + batch_size - 1, total_pages)


class DoclingParser(BaseParser):
    """Convert PDFs to Markdown while bounding each Docling result to a page range."""

    def __init__(self, page_batch_size: int | None = None):
        self.page_batch_size = (
            settings.DOCLING_PAGE_BATCH_SIZE if page_batch_size is None else page_batch_size
        )
        if self.page_batch_size < 1:
            raise ValueError("Docling 페이지 배치 크기는 1 이상이어야 합니다.")

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.layout_batch_size = 1
            pipeline_options.table_batch_size = 1
            pipeline_options.ocr_batch_size = 1
            pipeline_options.queue_max_size = 4

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info(
                "Docling DocumentConverter가 초기화되었습니다 "
                "(OCR: Disabled, Table: True, Page batch: %d).",
                self.page_batch_size,
            )
        except Exception as error:
            logger.error("Docling 초기화 실패: %s", error)
            self.converter = None

    @staticmethod
    def _page_count(pdf_path: str) -> int:
        with pymupdf.open(pdf_path) as pdf:
            return pdf.page_count

    @staticmethod
    def _export_without_headers_and_footers(doc) -> str:
        for item, _ in doc.iterate_items():
            if hasattr(item, "text"):
                stripped_text = item.text.strip()
                log_text = stripped_text[:50] + ("..." if len(stripped_text) > 50 else "")
                logger.debug("Docling Item: Label=%s, Text='%s'", item.label, log_text)
            else:
                logger.debug("Docling Item: Label=%s, No text attribute", item.label)

            if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER) and hasattr(item, "text"):
                logger.info("헤더/푸터 제거됨 (%s): %s", item.label, item.text.strip())
                item.text = ""

        return doc.export_to_markdown().strip()

    def parse(self, content: bytes, filename: str = None) -> List[ParsedDocument]:
        """Convert one queued PDF task in sequential page ranges and merge its Markdown."""
        if not self.converter:
            raise ImportError(
                "docling 패키지가 설치되지 않았거나 초기화에 실패했습니다. "
                "pip install docling로 설치하세요."
            )

        logger.info("Docling을 사용하여 파싱을 시도합니다: 파일=%s", filename)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            total_pages = self._page_count(tmp_path)
            ranges = list(page_ranges(total_pages, self.page_batch_size))
            markdown_parts = []

            for batch_number, (start_page, end_page) in enumerate(ranges, start=1):
                result = None
                doc = None
                try:
                    logger.info(
                        "Docling 페이지 배치 파싱: %d/%d, 페이지=%d-%d",
                        batch_number,
                        len(ranges),
                        start_page,
                        end_page,
                    )
                    result = self.converter.convert(
                        tmp_path,
                        page_range=(start_page, end_page),
                    )
                    doc = result.document
                    part_markdown = self._export_without_headers_and_footers(doc)
                    if part_markdown:
                        markdown_parts.append(part_markdown)
                    else:
                        logger.warning(
                            "Docling 페이지 배치가 빈 결과를 반환했습니다: 파일=%s, 페이지=%d-%d",
                            filename,
                            start_page,
                            end_page,
                        )
                finally:
                    # Retain only the compact Markdown string before converting the next range.
                    doc = None
                    result = None
                    gc.collect()

            markdown_text = "\n\n".join(markdown_parts).strip()
            if not markdown_text:
                raise ValueError(
                    "PDF에서 텍스트를 추출할 수 없습니다. "
                    "PDF가 텍스트 레이어를 포함하지 않거나 이미지로만 구성되어 있을 수 있습니다."
                )

            logger.info(
                "Docling 파싱 성공: 파일=%s, 페이지=%d, 페이지 배치=%d, 텍스트 길이=%d",
                filename,
                total_pages,
                self.page_batch_size,
                len(markdown_text),
            )
            return [
                ParsedDocument(
                    text=markdown_text,
                    metadata={
                        "filename": filename or "unknown.pdf",
                        "format": "markdown",
                        "parser": "docling",
                        "page_count": total_pages,
                        "page_batch_size": self.page_batch_size,
                    },
                )
            ]
        except Exception as error:
            logger.exception(
                "Docling 파싱 실패: %s, 파일=%s, 에러 타입=%s",
                error,
                filename,
                type(error).__name__,
            )
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def supported_extensions(self) -> List[str]:
        return ["pdf"]
