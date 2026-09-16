import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort




def compute_hist_embedding(crop,  bins=(8, 8, 8)):
    if crop.size == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)

class Tracker:

    def __init__(self, max_age=30, alpha=0.3, max_cosine_distance=0.2):
        self.tracker = DeepSort(max_age=max_age, embedder=None, max_cosine_distance=max_cosine_distance)
        self.alpha = alpha
        self.smoothed_boxes = {}



    def update(self, detections, frame):
        raw = []
        embeds = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1,  y2 - y1

            if w <= 0 or h <= 0:
                continue

            ltwh = [x1, y1, w, h] #left, top, width, high
            raw.append((ltwh, det["conf"], det["class_id"]))

            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            crop = frame[max(0, y1i):y2i, max(0, x1i):x2i]
            embeds.append(compute_hist_embedding(crop))

        tracks = self.tracker.update_tracks(raw, embeds=embeds)

        tracked = []
        current_id = set()


        for t in tracks:
            if not t.is_confirmed():
                continue # if confidence is too low

            track_id = t.track_id
            current_id.add(track_id)

            raw_box = t.to_ltrb()

            if track_id in self.smoothed_boxes:
                old_bbox = self.smoothed_boxes[track_id]

                smooth_bbox = [
                    (self.alpha * raw_box[i]) + ((1.0 - self.alpha) * old_bbox[i])
                    for i in range(4)
                ]
            else:
                smooth_bbox = list(raw_box)

            self.smoothed_boxes[track_id] = smooth_bbox

            tracked.append({
                "bbox": smooth_bbox,
                "raw_bbox": list(raw_box),
                "class_id": t.get_det_class(),
                "track_id": t.track_id,
            })

        self.smoothed_boxes = {
            tid: bbox for tid , bbox in self.smoothed_boxes.items() if tid in current_id
        }

        return tracked