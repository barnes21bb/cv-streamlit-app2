from typing import Dict, List, Tuple

import cv2

try:
    from ultralytics import YOLO
except Exception as e:  # ultralytics may not be installed when importing
    YOLO = None


class YOLODetector:
    """Wrapper around YOLOv8 model."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25):
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLOv8 detection")
        self.model = YOLO(model_name)
        self.conf = conf

    def set_conf(self, conf: float) -> None:
        self.conf = conf

    def detect_frame(self, frame) -> List[Dict]:
        """Run detection on a single frame and return list of detections."""
        results = self.model(frame, conf=self.conf)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            detections.append({
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
            })
        return detections

    def detect_video(self, path: str) -> Tuple[Dict[int, List[Dict]], Dict[int, int]]:
        """Run detection on a video and return annotations and counts per frame."""
        cap = cv2.VideoCapture(path)
        annotations = {}
        counts = {}
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detect_frame(frame)
            annotations[frame_num] = detections
            counts[frame_num] = len(detections)
            frame_num += 1
        cap.release()
        return annotations, counts
