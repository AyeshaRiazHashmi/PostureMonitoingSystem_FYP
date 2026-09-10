from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database import get_db
from models import PostureDetection, PostureSession, User
from routes.auth import get_current_user
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/analytics/summary")
async def get_summary(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    start_date = datetime.utcnow() - timedelta(days=days)
    
    detections = db.query(PostureDetection).filter(
        PostureDetection.timestamp >= start_date,
        PostureDetection.user_id == current_user.id
    ).all()
    
    total = len(detections)
    good = sum(1 for d in detections if d.posture_class == "good")
    bad = sum(1 for d in detections if d.posture_class == "bad")
    avg_confidence = sum(d.confidence for d in detections) / total if total > 0 else 0
    
    return {
        "period_days": days,
        "total_detections": total,
        "good_posture": good,
        "bad_posture": bad,
        "good_percentage": (good / total * 100) if total > 0 else 0,
        "bad_percentage": (bad / total * 100) if total > 0 else 0,
        "average_confidence": avg_confidence
    }

@router.get("/analytics/recent")
async def get_recent_detections(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    detections = db.query(PostureDetection).filter(
        PostureDetection.user_id == current_user.id
    ).order_by(desc(PostureDetection.timestamp)).limit(limit).all()
    
    return [{
        "id": d.id,
        "posture_class": d.posture_class,
        "confidence": d.confidence,
        "timestamp": d.timestamp,
        "detection_type": d.detection_type
    } for d in detections]

@router.get("/analytics/sessions")
async def get_sessions(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    sessions = db.query(PostureSession).filter(
        PostureSession.user_id == current_user.id
    ).order_by(desc(PostureSession.start_time)).limit(limit).all()
    
    return [{
        "session_id": s.session_id,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "is_active": s.is_active,
        "total_detections": s.total_detections,
        "good_posture_count": s.good_posture_count,
        "bad_posture_count": s.bad_posture_count,
        "detection_mode": s.detection_mode
    } for s in sessions]