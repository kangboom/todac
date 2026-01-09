"""
PDF 파서 관리 서비스
"""
from typing import Optional, Type, Dict, List
from functools import lru_cache
import logging

from app.services.parsers.base import BaseParser
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.core.config import settings

logger = logging.getLogger(__name__)

# 파서 클래스 매핑
PARSER_CLASSES: Dict[str, Type[BaseParser]] = {
    "llamaparse": LlamaParseParser,
    "pymupdf": PyMuPDFParser,
    "docling": DoclingParser,
}

# 폴백 순서 정의 (설정된 파서 실패 시 시도할 순서)
# Docling은 무거울 수 있으므로 기본 폴백 목록에서는 제외하거나 가장 마지막에 고려
FALLBACK_ORDER: List[str] = ["pymupdf", "llamaparse"]

@lru_cache(maxsize=1)
def get_active_parser() -> Optional[BaseParser]:
    """
    설정에 따른 활성 파서를 초기화하고 반환 (싱글톤 패턴, LRU Cache 사용)
    설정된 파서 초기화 실패 시 폴백 로직을 수행합니다.
    """
    target_parser_name = settings.PDF_PARSER.lower()
    
    # 1. 시도할 파서 목록 생성 (설정된 파서 -> 폴백 파서들)
    # 중복 제거를 위해 리스트 컴프리헨션 사용
    candidate_parsers = [target_parser_name] + [p for p in FALLBACK_ORDER if p != target_parser_name]
    
    for parser_name in candidate_parsers:
        parser_cls = PARSER_CLASSES.get(parser_name)
        if not parser_cls:
            logger.warning(f"알 수 없는 파서 타입: {parser_name}")
            continue
            
        try:
            logger.info(f"{parser_name} 파서 초기화 시도...")
            parser_instance = parser_cls()
            logger.info(f"{parser_name} 파서가 성공적으로 초기화되었습니다.")
            return parser_instance
        except Exception as e:
            logger.warning(f"{parser_name} 파서 초기화 실패: {e}")
            continue
            
    logger.error("모든 파서 초기화에 실패했습니다.")
    # 모든 파서 실패 시 예외를 발생시킬지, None을 반환할지 결정해야 함.
    # 기존 로직은 ValueError를 발생시켰으므로 유지
    raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지나 API 키 설정을 확인하세요.")

def get_parser(filename: str) -> Optional[BaseParser]:
    """
    파일명에 맞는 파서 반환
    현재는 모든 파서가 PDF를 처리하므로 활성화된 파서를 반환하고,
    확장자 체크만 수행합니다.
    """
    try:
        parser = get_active_parser()
        if parser and parser.can_parse(filename):
            return parser
    except ValueError as e:
        logger.error(f"파서 가져오기 실패: {e}")
        return None
        
    return None
