from deep_sort_realtime.deepsort_tracker import DeepSort



class Tracker:

    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)



    def update(self, detections, frame):
        raw = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            ltwh = [x1, y1, x2 - x1, y2 - y1] #left, top, width, high
            raw.append((ltwh, det["conf"], det["class_id"]))

        tracks = self.tracker.update_tracks(raw, frame=frame)

        tracked = []

        for t in tracks:
            if not t.is_confirmed():
                continue # if confidence is too low

            x1, y1, x2, y2 = t.to_ltrb()
            tracked.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": t.get_det_class(),
                "track_id": t.track_id,
            })

        return tracked