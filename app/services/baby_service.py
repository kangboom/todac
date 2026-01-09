"""
아기 프로필 서비스
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.baby import BabyProfile
from app.dto.baby import BabyCreateRequest, BabyUpdateRequest
from typing import List
import uuid


def create_baby(db: Session, user_id: uuid.UUID, request: BabyCreateRequest) -> BabyProfile:
    """아기 프로필 생성"""
    new_baby = BabyProfile(
        user_id=user_id,
        name=request.name,
        birth_date=request.birth_date,
        due_date=request.due_date,
        gender=request.gender,
        birth_weight=request.birth_weight,
        medical_history=request.medical_history
    )
    
    db.add(new_baby)
    db.commit()
    db.refresh(new_baby)
    
    return new_baby


def get_baby_by_id(db: Session, baby_id: uuid.UUID, user_id: uuid.UUID) -> BabyProfile:
    """아기 프로필 조회 (본인 것만)"""
    baby = db.query(BabyProfile).filter(
        BabyProfile.id == baby_id,
        BabyProfile.user_id == user_id
    ).first()
    
    if not baby:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="아기 프로필을 찾을 수 없습니다."
        )
    
    return baby


def get_babies_by_user(db: Session, user_id: uuid.UUID) -> List[BabyProfile]:
    """사용자의 모든 아기 프로필 조회"""
    babies = db.query(BabyProfile).filter(
        BabyProfile.user_id == user_id
    ).order_by(BabyProfile.created_at.desc()).all()
    
    return babies


def update_baby(
    db: Session, 
    baby_id: uuid.UUID, 
    user_id: uuid.UUID, 
    request: BabyUpdateRequest
) -> BabyProfile:
    """아기 프로필 수정"""
    baby = get_baby_by_id(db, baby_id, user_id)
    
    # 업데이트할 필드만 수정
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(baby, field, value)
    
    db.commit()
    db.refresh(baby)
    
    return baby


def delete_baby(db: Session, baby_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """아기 프로필 삭제"""
    baby = get_baby_by_id(db, baby_id, user_id)
    
    db.delete(baby)
    db.commit()

