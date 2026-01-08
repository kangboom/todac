"""
업로드 문서 관리 테이블 (KnowledgeDoc)
"""
from sqlalchemy import Column, String, Text, Index, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class KnowledgeDoc(Base):
    """업로드 문서 관리 테이블 - S3에 저장된 문서와 Milvus(벡터 DB)와 동기화되는 메타데이터"""
    __tablename__ = "knowledge_docs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="문서의 고유 식별자 (자동 생성)")
    filename = Column(Text, nullable=False, comment="원본 파일의 이름 (확장자 포함)")
    storage_url = Column(Text, nullable=False, unique=True, comment="S3에 저장된 Markdown 파일 경로")
    raw_pdf_url = Column(Text, nullable=True, comment="S3에 저장된 원본 PDF 파일의 저장 경로")
    doc_hash = Column(String(64), nullable=True, index=True, comment="문서 내용의 해시값 (중복 업로드 방지)")
    file_size = Column(Integer, nullable=True, comment="파일 크기 (bytes)")
    meta_info = Column(JSONB, nullable=False, server_default='{}', comment="가변 메타데이터 (작성자, 태그, 카테고리 등)")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, comment="레코드 생성 일시")
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="레코드 최종 수정 일시")

    # 인덱스 추가
    __table_args__ = (
        Index('idx_knowledge_docs_doc_hash', 'doc_hash'),
    )

    def __repr__(self):
        return f"<KnowledgeDoc(id={self.id}, filename={self.filename}, storage_url={self.storage_url})>"

