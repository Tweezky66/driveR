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

    def __init__(self, max_age=30, alpha=0.3, max_cosine_distance=0.35, dublicate_iou=0.6, max_iou_distance=0.9):
        self.tracker = DeepSort(max_age=max_age, n_init=2, embedder=None, max_cosine_distance=max_cosine_distance, max_iou_distance=max_iou_distance)
        self.alpha = alpha
        self.dublicate_iou = dublicate_iou
        self.class_votes = {} # track_id -> {class_id: count}
        self.smoothed_boxes = {}



    def iou(self, box_a, box_b):
        # boxes are (x1, y1, x2, y2)
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h


        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0
        return intersection / union


    def vote_class(self, track_id, class_id):
        if track_id not in self.class_votes:
            self.class_votes[track_id] = {}
        votes = self.class_votes[track_id]

        if class_id is not None:
            votes[class_id] = 1 + votes.get(class_id, 0)

        if not votes:
            return class_id
        return max(votes, key=votes.get) # msot freaquent class first




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

        

        live = []
        for t in tracks:
            if not t.is_confirmed():
                continue # if confidence is too low
            if t.time_since_update > 3:
                continue # coasting: bbox is Kalman pred only
            live.append(t)

        # keep the long  living one
        live.sort(key=lambda t: (t.time_since_update, -t.age)) # add a tuple, make sure if time is equal, age is not

        tracked = []
        kept_boxes = []

        for t in live:
            raw_box = t.to_ltrb(orig=True) # detection match, not the calman filter match

            raw_box = list(raw_box)

            is_dublicate = False
            for kept in kept_boxes:
                if self.iou(raw_box, kept) > self.dublicate_iou:
                    is_dublicate = True
                    break
            if is_dublicate:
                continue
            kept_boxes.append(raw_box)

            track_id = t.track_id

            if track_id in self.smoothed_boxes:
                old = self.smoothed_boxes[track_id]
                smooth_box = []

                for i in range(4):
                    smooth_box.append(self.alpha * raw_box[i] + (1.0 - self.alpha) * old[i])
            else:
                smooth_box = raw_box

            self.smoothed_boxes[track_id] = smooth_box

            tracked.append({
                "bbox": smooth_box,
                "raw_bbox": raw_box,
                "matched": t.time_since_update == 0,
                "class_id": self.vote_class(track_id, t.get_det_class()),
                "track_id": t.track_id,
            })

        # forget only tracks which DeepSORT itself has deleted (coasting ones keep their own history)
        alive_ids = set()
        for t in tracks:
            alive_ids.add(t.track_id)

        for track_id in list(self.smoothed_boxes.keys()):
            if track_id not in alive_ids:
                del self.smoothed_boxes[track_id]
        for track_id in list(self.class_votes.keys()):
            if track_id not in alive_ids:
                del self.class_votes[track_id]

        return tracked