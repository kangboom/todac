"""
문서 파서 기본 클래스 (확장 가능한 구조)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """문서 파서 기본 클래스"""
    
    @abstractmethod
    def parse(self, content: bytes, filename: str = None) -> List[Dict[str, Any]]:
        """
        문서를 파싱하여 텍스트 청크 리스트 반환
        
        Args:
            content: 파일 바이너리 내용
            filename: 파일명 (선택)
        
        Returns:
            청크 리스트, 각 청크는 {"text": str, "metadata": dict} 형식
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 리스트"""
        pass
    
    def can_parse(self, filename: str) -> bool:
        """파일을 파싱할 수 있는지 확인"""
        if not filename:
            return False
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return ext in self.supported_extensions()

