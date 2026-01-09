"""
PDF 파서 관리 서비스
"""
from typing import List, Optional
from app.services.parsers.base import BaseParser
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# 파서 인스턴스 (설정에 따라 선택)
_parsers: List[BaseParser] = []


def initialize_parsers():
    """설정에 따라 파서 초기화"""
    global _parsers
    if _parsers:
        return  # 이미 초기화됨
    
    parser_type = settings.PDF_PARSER.lower()
    
    if parser_type == "llamaparse":
        try:
            _parsers.append(LlamaParseParser())
            logger.info("LlamaParse 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"LlamaParse 파서 초기화 실패: {str(e)}")
            # 폴백으로 PyMuPDF 시도
            try:
                _parsers.append(PyMuPDFParser())
                logger.warning("LlamaParse 파서 초기화 실패, PyMuPDF 파서로 폴백합니다.")
            except Exception as e2:
                logger.error(f"PyMuPDF 파서 초기화도 실패: {str(e2)}")
                raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
    
    elif parser_type == "pymupdf":
        try:
            _parsers.append(PyMuPDFParser())
            logger.info("PyMuPDF 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"PyMuPDF 파서 초기화 실패: {str(e)}")
            # 폴백으로 LlamaParse 시도 (API 키가 있는 경우)
            if settings.LLAMAPARSE_API_KEY:
                try:
                    _parsers.append(LlamaParseParser())
                    logger.warning("PyMuPDF 파서 초기화 실패, LlamaParse 파서로 폴백합니다.")
                except Exception as e2:
                    logger.error(f"LlamaParse 파서 초기화도 실패: {str(e2)}")
                    raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
            else:
                raise ValueError(
                    "PyMuPDF 파서를 초기화할 수 없습니다. "
                    "pymupdf와 pymupdf4llm 패키지가 설치되어 있는지 확인하거나, "
                    "LlamaParse를 사용하려면 LLAMAPARSE_API_KEY를 설정하세요."
                )
    
    elif parser_type == "docling":
        try:
            _parsers.append(DoclingParser())
            logger.info("Docling 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Docling 파서 초기화 실패: {str(e)}")
            # 폴백으로 PyMuPDF 시도
            try:
                _parsers.append(PyMuPDFParser())
                logger.warning("Docling 파서 초기화 실패, PyMuPDF 파서로 폴백합니다.")
            except Exception as e2:
                logger.error(f"PyMuPDF 파서 초기화도 실패: {str(e2)}")
                # 폴백으로 LlamaParse 시도 (API 키가 있는 경우)
                if settings.LLAMAPARSE_API_KEY:
                    try:
                        _parsers.append(LlamaParseParser())
                        logger.warning("PyMuPDF 파서 초기화도 실패, LlamaParse 파서로 폴백합니다.")
                    except Exception as e3:
                        logger.error(f"LlamaParse 파서 초기화도 실패: {str(e3)}")
                        raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
                else:
                    raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
    else:
        raise ValueError(
            f"지원하지 않는 PDF 파서 타입: {parser_type}. "
            f"'pymupdf', 'llamaparse', 또는 'docling'을 사용하세요."
        )


def get_parser(filename: str) -> Optional[BaseParser]:
    """파일명에 맞는 파서 찾기"""
    # 파서가 초기화되지 않았으면 초기화
    if not _parsers:
        initialize_parsers()
    
    for parser in _parsers:
        if parser.can_parse(filename):
            return parser
    return None

