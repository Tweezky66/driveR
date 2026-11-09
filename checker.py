import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import risk_engine_cpp  as _risk
from ultralytics import YOLO
from tqdm.auto import tqdm

from perception.Detector import Detector
from perception.tracker import Tracker
from sim.depth.dataset import IMAGENET_MEAN, IMAGENET_STD
from sim.depth.model import DepthNet



class VideoChecker:
    MODEL_W = 640
    MODEL_H = 352
    TRAIN_FOCAL = 320.0
    MAX_DEPTH = 50.0
    WARMUP = 10
    TTC_WARNING = 2.0

    HEIGHTS = {0: 1.7, 2: 1.5, 3: 1.4, 5: 3.2, 7: 3.0} # approx real word height in COCO classes
    VEHICLES = [2, 3, 5, 7]
    METHODS = ["pinhole", "custom", "yolo"]


    def __init__(self, root, video_name, hfov_deg):
        self.root = root
        self.video_path = root / "Datasets" / video_name
        self.hfov = np.radians(hfov_deg)       
        self.out_dir = root / "docs" / "checker"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

        self.session = ort.InferenceSession(str(root / "depth_model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        

        # models to compare
        self.detector = Detector(str(root / "yolo26s.pt"))
        self.tracker = Tracker()
        self.yolo_depth = YOLO(str(root / "sim" / "yolo26s-depth.pt"))
        
        self.model_focal = (self.MODEL_W / 2) / np.tan(self.hfov / 2)
        self.depth_scale = self.model_focal / self.TRAIN_FOCAL

        self.fps = 60
        self.rows = []
        self.times = {"custom": [], "yolo": []}
        self.filters ={}
        self.last_time = {}

    def check_parity(self):
        model = DepthNet(freeze_encoder=False)
        weights = torch.load(self.root / "new_best_depth.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        model.eval()

        dummy  = np.random.randn(1, 3, self.MODEL_H, self.MODEL_W).astype(np.float32)


        with torch.no_grad():
            torch_out = model(torch.from_numpy(dummy)).numpy()
        onnx_out = self.session.run(None, {self.input_name: dummy})[0]

        diff = np.abs(onnx_out - torch_out).max()

        print(f"ONNX vs torch difference: {diff:.2e}")



    def predict_custom(self, small):
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0 # normilizing
        x = x.transpose(2, 0, 1) # HWC - > CHW
        x = (x - self.mean) / self.std
        x = x[np.newaxis] # analog for unsqueeze(0) but  np

        start = time.perf_counter()
        log_depth = self.session.run(None, {self.input_name: x})[0]
        self.times["custom"].append(time.perf_counter() - start)

        log_depth = np.minimum(log_depth[0, 0], 7.0)
        return np.exp(log_depth)  * self.depth_scale


    def predict_yolo(self, small):
        start = time.perf_counter()
        result = self.yolo_depth.predict(small, imgsz=768, device="cpu", verbose=False)[0]
        depth = result.depth.data.cpu().numpy()
        self.times["yolo"].append(time.perf_counter() - start)

        depth = np.squeeze(depth)
        if depth.shape != (self.MODEL_H, self.MODEL_W):
            depth = cv2.resize(depth, (self.MODEL_W, self.MODEL_H))
        return depth



    def pinhole_distance(self, frame_width, box_height,  class_id):
        if box_height < 5:
            return np.nan
        focal = (frame_width / 2) / np.tan(self.hfov / 2)
        return focal * self.HEIGHTS[class_id] / box_height


    def box_distance(self, depth, box):
        x1, y1,  x2, y2 = box
        w  = x2 - x1
        h = y2 - y1


        left = int(x1 + 0.25 * w)
        right = int(x2 - 0.25 * w)
        top = int(y1 + 0.3 * h)
        bottom = int(y2 - 0.1 * h)

        core = depth[top:bottom, left:right]
        if core.size < 10:
            return np.nan

        return float(np.median(core))


    def smooth(self, track_id, method, distance, now):
        key = (track_id, method)
        if np.isnan(distance):
            return np.nan, np.nan

        if key not in self.filters:
            self.filters[key]  = _risk.KalmanFilter()
            self.last_time[key] = now
            return distance, np.nan

        dt = now - self.last_time[key]
        self.last_time[key] = now
        kf = self.filters[key]
        kf.predict(dt)

        r_z = 1.0 + 0.05 * distance ** 2  # same noise model as risk_manager
        x_smooth, z_smooth = kf.update(0.0, distance, 1.0,  r_z)
        closing = -kf.velocity_z()
        return z_smooth, closing

    def procces_frame(self, frame,  index):
        frame_h, frame_w = frame.shape[:2]
        small = cv2.resize(frame, (self.MODEL_W, self.MODEL_H))

        custom_depth = self.predict_custom(small)
        yolo_depth = self.predict_yolo(small)


        result = self.detector.predict(frame, classes=list(self.HEIGHTS.keys()))
        detections = self.detector.extract_detections(result)
        tracks = self.tracker.update(detections, frame)


        sx = self.MODEL_W / frame_w
        sy = self.MODEL_H / frame_h

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            class_id = int(track["class_id"])

            # the same  box but scaling to (640, 352) size
            bx1 = max(0, x1 * sx)
            by1 = max(0, y1 * sy)
            bx2 = min(self.MODEL_W,  x2 * sx)
            by2 = min(self.MODEL_H,  y2 * sy)

            box = (bx1, by1, bx2, by2)

            now = index / self.fps
            row = {
                "frame": index,
                "time": now,
                "track_id": track["track_id"],
                "class_id": class_id,
                "center_x": (x1 + x2) / 2 / frame_w,
                "bottom": y2 / frame_h,
                "pinhole": self.pinhole_distance(frame_w, y2 - y1, class_id),
                "custom": self.box_distance(custom_depth, box),
                "yolo": self.box_distance(yolo_depth, box),
            }


            for method in  self.METHODS:
                z_smooth, closing = self.smooth(track["track_id"], method, row[method],  now)
                row[method + "_smooth"] = z_smooth
                row[method + "_closing"] = closing
            
            self.rows.append(row)

            label = f"P{row['pinhole']:.0f} C{row['custom']:.0f} Y{row['yolo']:.0f}"
            cv2.rectangle(small, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 1)
            cv2.putText(small, label, (int(bx1), int(by1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return small, custom_depth, yolo_depth


    def colorize(self, depth):
        depth = np.clip(depth, 0, self.MAX_DEPTH)
        gray = 255 - (depth / self.MAX_DEPTH * 255).astype(np.uint8) # normilizing and scaling to get near=brighter
        return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = self.out_dir / f"{self.video_path.stem}_compare.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (3 * self.MODEL_W, self.MODEL_H))


        index = 0

        bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="checking")

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            small, custom_depth, yolo_depth = self.procces_frame(frame, index)
            
            panel = np.hstack([small, self.colorize(custom_depth), self.colorize(yolo_depth)])
            writer.write(panel)

            
            index += 1
            bar.update(1)


        bar.close()
        cap.release()
        writer.release()

        self.save_csv()
        self.report()
        self.plot_lead()



    def save_csv(self):
        path = self.out_dir / f"{self.video_path.stem}_rows.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)



    def jitter(self, method):

        per_track = {}
        for row in self.rows:
            track_id = row["track_id"]
            if track_id not in per_track:
                per_track[track_id] = []
            per_track[track_id].append(row[method])


        changes = []
        for values in per_track.values():
            for i in range(1, len(values)):
                a = values[i - 1]
                b = values[i]
                if np.isnan(a) or np.isnan(b):
                    continue

                changes.append(abs(b - a) / b)

        return np.median(changes)


    def agreement(self, method_a, method_b):
        ratios = []
        for row in self.rows:
            a = row[method_a]
            b = row[method_b]
            if np.isnan(a) or np.isnan(b):
                continue
            ratios.append(a / b)

        ratios = np.array(ratios)
        scale = np.median(ratios) # howw muc hreally a (in times) bigger than b
        spread = np.median(np.abs(ratios / scale - 1))
        return scale, spread

    def report(self):
        print("\nLatency, median ms (CPU)")
        for name in self.times:
            values = np.array(self.times[name][self.WARMUP:]) * 1000
            print(f"  {name}: {np.median(values):.1f}")

        print("\nJitter (lower = steadier)")
        for method in self.METHODS:
            print(f"  {method}: {100 * self.jitter(method):.1f} %")

        print("\nAgreement: scale, spread after scale removed")
        pairs = [("custom", "pinhole"), ("yolo", "pinhole"), ("custom", "yolo")]
        for a, b in pairs:
            scale, spread = self.agreement(a, b)
            print(f"  {a} vs {b}: x{scale:.2f}, {100 * spread:.1f} %")


    
    def plot_lead(self):
        lead = {}

        collision_timing = 6
        gap = 1
        
        for row in self.rows:
            if row["class_id"] not in self.VEHICLES:
                continue
            if row["center_x"] < 0.35 or row["center_x"] > 0.65:
                continue

            frame = row["frame"]
            if frame not in lead or row["bottom"] > lead[frame]["bottom"]:
                lead[frame] = row

        frames = sorted(lead.keys())
        times = []
        for frame in frames:
            times.append(lead[frame]["time"])

        fig, (ax_dist, ax_ttc) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        print(f"\nFirst TTC below {self.TTC_WARNING} s")

        for method in self.METHODS:
            raw, smooth, ttc = [], [], []
            for frame in frames:
                lead_row = lead[frame]
                raw.append(lead_row[method])
                smooth.append(lead_row[method + "_smooth"])
                closing = lead_row[method + "_closing"]
                if closing > 0.1:
                    ttc.append(lead_row[method + "_smooth"] / closing)
                else:
                    ttc.append(np.nan)

            line = ax_dist.plot(times, smooth, label=method)[0]
            ax_dist.plot(times, raw, color=line.get_color(), alpha=0.3, linewidth=0.8)
            ax_ttc.plot(times, ttc, color=line.get_color(), label=method)

            first = "never"
            for t, value in zip(times, ttc):
                if value < self.TTC_WARNING:
                    first = f"{t:.2f} s"
                    break
            print(f"  {method}: {first}")

        ax_dist.set_ylabel("lead distance [m] (faded = raw)")
        ax_dist.legend()
        ax_ttc.axhline(self.TTC_WARNING, color="red", linestyle="--")
        ax_ttc.axvline(collision_timing - gap, color="red",  linestyle="-", label="The collision gap (video) ")
        ax_ttc.axvline(collision_timing + gap, color="red",  linestyle="-")
        ax_ttc.set_ylim(0, 10)
        ax_ttc.set_ylabel("TTC [s] (Kalman)")
        ax_ttc.set_xlabel("Time [s]")
        ax_ttc.legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / f"{self.video_path.stem}_lead.png")
        plt.close(fig)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    checker = VideoChecker(root, "brake_check.mp4", hfov_deg=70)
    checker.check_parity()
    checker.run()