"""
S3 스토리지 서비스
"""
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from app.core.config import settings
import logging
import uuid
from typing import Dict
from functools import lru_cache
from app.dto.knowledge import StoragePath

logger = logging.getLogger(__name__)


@lru_cache()
def get_s3_client():
    """S3 클라이언트 싱글톤 (LRU Cache 사용)"""
    s3_config = {
        'service_name': 's3',
        'region_name': settings.S3_REGION,
    }
    
    # AWS 자격 증명이 있으면 사용
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        s3_config['aws_access_key_id'] = settings.AWS_ACCESS_KEY_ID
        s3_config['aws_secret_access_key'] = settings.AWS_SECRET_ACCESS_KEY
    
    # MinIO 등 로컬 S3 호환 서비스 사용 시
    if settings.S3_ENDPOINT_URL:
        s3_config['endpoint_url'] = settings.S3_ENDPOINT_URL
    
    return boto3.client(**s3_config)


def upload_to_s3(content: bytes, s3_key: str, content_type: str = None) -> str:
    """
    S3에 파일 업로드
    
    Args:
        content: 파일 내용 (bytes)
        s3_key: S3 객체 키 (경로)
        content_type: Content-Type (예: 'application/pdf', 'text/markdown')
    
    Returns:
        S3 URL (s3://bucket-name/key 형식)
    """
    try:
        s3_client = get_s3_client()
        bucket_name = settings.S3_BUCKET_NAME
        
        # 업로드 옵션
        upload_kwargs = {
            'Bucket': bucket_name,
            'Key': s3_key,
            'Body': content,
        }
        
        # Content-Type 설정
        if content_type:
            upload_kwargs['ContentType'] = content_type
        else:
            # 파일 확장자로 자동 판단
            if s3_key.endswith('.pdf'):
                upload_kwargs['ContentType'] = 'application/pdf'
            elif s3_key.endswith('.md'):
                upload_kwargs['ContentType'] = 'text/markdown'
            elif s3_key.endswith('.png'):
                upload_kwargs['ContentType'] = 'image/png'
            elif s3_key.endswith('.jpg') or s3_key.endswith('.jpeg'):
                upload_kwargs['ContentType'] = 'image/jpeg'
        
        # S3에 업로드
        s3_client.put_object(**upload_kwargs)
        
        s3_url = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"S3 업로드 완료: {s3_url}")
        
        return s3_url
        
    except ClientError as e:
        logger.error(f"버킷 이름: {bucket_name}")
        logger.error(f"S3 업로드 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 업로드 중 오류가 발생했습니다: {str(e)}"
        )


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    S3 URL에서 bucket과 key 추출
    
    Args:
        s3_url: S3 URL (s3://bucket-name/key 형식)
    
    Returns:
        (bucket_name, key) 튜플
    """
    if not s3_url.startswith('s3://'):
        raise ValueError(f"유효하지 않은 S3 URL 형식: {s3_url}")
    
    # s3:// 제거
    path = s3_url[5:]
    
    # 첫 번째 '/'를 기준으로 bucket과 key 분리
    parts = path.split('/', 1)
    bucket_name = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket_name, key


def delete_from_s3(s3_url: str) -> None:
    """
    S3에서 파일 삭제
    
    Args:
        s3_url: S3 URL (s3://bucket-name/key 형식)
    """
    try:
        if not s3_url:
            logger.warning("S3 URL이 비어있어 삭제를 건너뜁니다.")
            return
        
        s3_client = get_s3_client()
        bucket_name, key = parse_s3_url(s3_url)
        
        # S3에서 파일 삭제
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        
        logger.info(f"S3 파일 삭제 완료: {s3_url}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.warning(f"S3 파일이 이미 존재하지 않습니다: {s3_url}")
        else:
            logger.error(f"S3 파일 삭제 실패: {str(e)}, URL: {s3_url}")
            # S3 삭제 실패해도 계속 진행 (PostgreSQL, Milvus 삭제는 진행)
    except Exception as e:
        logger.error(f"S3 파일 삭제 중 예외 발생: {str(e)}, URL: {s3_url}")
        # 예외 발생해도 계속 진행


def generate_storage_paths(doc_id: uuid.UUID, original_filename: str) -> StoragePath:
    """
    S3 저장 경로 생성 (새로운 구조)
    
    Args:
        doc_id: 문서 ID
        original_filename: 원본 파일명
    
    Returns:
        StoragePath 객체
        {
            'raw_pdf_key': 'dev/raw/doc_12345/original.pdf' (dev) 또는 'prod/raw/doc_12345/original.pdf' (prod)
            'processed_md_key': 'dev/processed/doc_12345/content.md'
            'images_dir': 'dev/processed/doc_12345/images/'
        }
    """
    doc_dir = f"doc_{doc_id}"
    
    # 환경별 최상위 폴더 설정
    # 개발환경(development) -> 'dev', 배포환경(production) -> 'prod'
    env_folder = "prod" if settings.ENVIRONMENT == "production" else "dev"
    
    # 원본 파일명에서 확장자 제거
    base_filename = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    
    return StoragePath(
        raw_pdf_key=f"{env_folder}/raw/{doc_dir}/{original_filename}",
        processed_md_key=f"{env_folder}/processed/{doc_dir}/{base_filename}.md",
        images_dir=f"{env_folder}/processed/{doc_dir}/images/"
    )

