from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from database import get_db
from models import PostureDetection, PostureSession, User
from extensions import detector
from routes.auth import get_current_user
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
import uuid
import os
import base64

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    session_id: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    detections = detector.detect(image)
    annotated_image = detector.draw_detections(image, detections)
    
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    
    for det in detections:
        detection_record = PostureDetection(
            user_id=current_user.id,
            posture_class=det["class"],
            confidence=det["confidence"],
            bbox_x1=det["bbox"]["x1"],
            bbox_y1=det["bbox"]["y1"],
            bbox_x2=det["bbox"]["x2"],
            bbox_y2=det["bbox"]["y2"],
            detection_type="image"
        )
        db.add(detection_record)
    
    if session_id:
        session = db.query(PostureSession).filter(
            PostureSession.session_id == session_id,
            PostureSession.user_id == current_user.id
        ).first()
        if session and session.is_active:
            session.total_detections += len(detections)
            session.good_posture_count += sum(1 for d in detections if d["class"] == "good")
            session.bad_posture_count += sum(1 for d in detections if d["class"] == "bad")
    
    db.commit()
    
    return {
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
    }

@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    video_filename = f"{uuid.uuid4()}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    
    with open(video_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    
    output_filename = f"processed_{video_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    result = detector.process_video(video_path, output_path)
    
    for i in range(result["good_posture"]):
        detection_record = PostureDetection(
            user_id=current_user.id,
            posture_class="good",
            confidence=0.85,
            detection_type="video"
        )
        db.add(detection_record)
    
    for i in range(result["bad_posture"]):
        detection_record = PostureDetection(
            user_id=current_user.id,
            posture_class="bad",
            confidence=0.85,
            detection_type="video"
        )
        db.add(detection_record)
    
    db.commit()
    
    return {
        "status": "success",
        "result": result,
        "download_url": f"/api/download/video/{output_filename}"
    }

@router.get("/download/video/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@router.websocket("/detect/live")
async def detect_live(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                detections = detector.detect(image)
                annotated_image = detector.draw_detections(image, detections)
                
                _, buffer = cv2.imencode('.jpg', annotated_image)
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "detections": detections,
                    "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
                })
    except WebSocketDisconnect:
        print("Client disconnected")

@router.post("/session/start")
async def start_session(
    mode: str = "image",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session_id = str(uuid.uuid4())
    session = PostureSession(
        user_id=current_user.id,
        session_id=session_id,
        detection_mode=mode
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session_id, "start_time": session.start_time}

@router.post("/session/end/{session_id}")
async def end_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session = db.query(PostureSession).filter(
        PostureSession.session_id == session_id,
        PostureSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.end_time = datetime.utcnow()
    session.is_active = False
    db.commit()
    
    return {
        "session_id": session_id,
        "duration": (session.end_time - session.start_time).total_seconds(),
        "total_detections": session.total_detections,
        "good_posture_count": session.good_posture_count,
        "bad_posture_count": session.bad_posture_count,
        "avg_confidence": session.avg_confidence
    }