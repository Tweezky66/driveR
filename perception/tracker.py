from deep_sort_realtime.deepsort_tracker import DeepSort



class Tracker:

    def __init__(self, max_age=30, alpha=0.3):
        self.tracker = DeepSort(max_age=max_age)
        self.alpha = alpha
        self.smoothed_boxes = {}



    def update(self, detections, frame):
        raw = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            ltwh = [x1, y1, x2 - x1, y2 - y1] #left, top, width, high
            raw.append((ltwh, det["conf"], det["class_id"]))

        tracks = self.tracker.update_tracks(raw, frame=frame)

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
                "class_id": t.get_det_class(),
                "track_id": t.track_id,
            })

        self.smoothed_boxes = {
            tid: bbox for tid , bbox in self.smoothed_boxes.items() if tid in current_id
        }

        return tracked