"""
사용자 및 아기 프로필 관련 API
"""
from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.dto.baby import BabyCreateRequest, BabyUpdateRequest, BabyResponse
from app.dto.auth import UserResponse
from app.services import baby_service

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


@router.post(
    "/babies", 
    response_model=BabyResponse, 
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(oauth2_scheme)]
)
def create_baby(
    request: BabyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    아기 프로필 등록
    
    - **name**: 아기 이름/태명
    - **birth_date**: 실제 태어난 날 (예방접종 기준)
    - **due_date**: 출산 예정일 (교정 연령/발달 평가 기준)
    - **gender**: 성별 (M 또는 F, 선택)
    - **birth_weight**: 출생 체중 (kg)
    - **medical_history**: 기저질환 리스트 (선택)
    """
    baby = baby_service.create_baby(db, current_user.id, request)
    return BabyResponse.model_validate(baby)


@router.get(
    "/babies", 
    response_model=List[BabyResponse],
    dependencies=[Depends(oauth2_scheme)]
)
def get_my_babies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """내 아기 프로필 목록 조회"""
    babies = baby_service.get_babies_by_user(db, current_user.id)
    return [BabyResponse.model_validate(baby) for baby in babies]


@router.get(
    "/babies/{baby_id}", 
    response_model=BabyResponse,
    dependencies=[Depends(oauth2_scheme)]
)
def get_baby(
    baby_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """아기 프로필 상세 조회"""
    baby = baby_service.get_baby_by_id(db, uuid.UUID(baby_id), current_user.id)
    return BabyResponse.model_validate(baby)


@router.put(
    "/babies/{baby_id}", 
    response_model=BabyResponse,
    dependencies=[Depends(oauth2_scheme)]
)
def update_baby(
    baby_id: str,
    request: BabyUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """아기 프로필 수정"""
    baby = baby_service.update_baby(db, uuid.UUID(baby_id), current_user.id, request)
    return BabyResponse.model_validate(baby)


@router.delete(
    "/babies/{baby_id}", 
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(oauth2_scheme)]
)
def delete_baby(
    baby_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """아기 프로필 삭제"""
    baby_service.delete_baby(db, uuid.UUID(baby_id), current_user.id)
    return None


@router.get(
    "/me", 
    response_model=UserResponse,
    dependencies=[Depends(oauth2_scheme)]
)
def get_me(current_user: User = Depends(get_current_user)):
    """현재 로그인한 사용자 정보 조회"""
    return UserResponse.model_validate(current_user)
