from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Dict
import os

class PostureDetector:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)
        self.class_names = {0: "bad", 1: "good"}
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, "unknown")
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img_copy = image.copy()
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            class_name = det["class"]
            confidence = det["confidence"]
            
            color = (0, 255, 0) if class_name == "good" else (0, 0, 255)
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            
            label = f"{class_name.upper()}: {confidence:.2%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_copy, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
            cv2.putText(img_copy, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img_copy
    
    def process_video(self, video_path: str, output_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame)
            all_detections.extend(detections)
            
            annotated_frame = self.draw_detections(frame, detections)
            out.write(annotated_frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        good_count = sum(1 for d in all_detections if d["class"] == "good")
        bad_count = sum(1 for d in all_detections if d["class"] == "bad")
        
        return {
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "total_detections": len(all_detections),
            "good_posture": good_count,
            "bad_posture": bad_count,
            "output_path": output_path
        }

detector = PostureDetector()