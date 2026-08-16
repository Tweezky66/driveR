from ultralytics import YOLO

# COCO class ids: 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck, 9 traffic light, 11 stop sign

DEFAULT_CLASSES = [0, 2, 3, 5, 7, 9, 11]

class Detector:
    def __init__(self, model_name="yolov8s.pt"):
        self.model = YOLO(model_name)

    def predict(self, frame, conf=0.30, classes=None):
        results = self.model(
            frame,
            conf=conf,
            classes=classes
        )

        return results[0]

    @staticmethod # use static to call func without creating an object
    def extract_detections(result):
        detections = []

        for box in result.boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "class_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
            })
        return detections