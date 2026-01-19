from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class OfficialQnA(Base):
    """Q&A 테이블"""
    __tablename__ = "official_qna"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, nullable=False, index=True, comment="카테고리 (예: 예방접종, 영양, 응급)")
    question = Column(String, nullable=False, comment="질문 (Vector Indexing 대상)")
    answer = Column(Text, nullable=False, comment="공식 답변")
    source = Column(String, nullable=False, comment="출처 (예: 대한신생아학회)")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
