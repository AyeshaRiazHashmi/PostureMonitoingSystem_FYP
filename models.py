from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    sessions = relationship("PostureSession", back_populates="user")
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

class PostureDetection(Base):
    __tablename__ = "posture_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(Integer, ForeignKey("posture_sessions.id"), nullable=True)
    image_path = Column(String, nullable=True)
    posture_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    detection_type = Column(String, default="image")
    
class PostureSession(Base):
    __tablename__ = "posture_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_detections = Column(Integer, default=0)
    good_posture_count = Column(Integer, default=0)
    bad_posture_count = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    detection_mode = Column(String, default="image")
    
    user = relationship("User", back_populates="sessions")