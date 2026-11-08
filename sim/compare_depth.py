from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from ultralytics import YOLO
from perception.Detector import Detector

from sim.depth.dataset import IMAGENET_MEAN, IMAGENET_STD
from sim.depth.model import DepthNet


class DepthEvaluator:
    EPISODES = [
        "Town01_ClearNoon_s101",
        "Town10HD_Opt_CloudyNoon_s110",
    ]

    BANDS = [
        ("0-4 m", 0, 4),
        ("4-8 m", 4, 8),
        ("8-10 m", 8, 10),
    ]

    MODELS = ["Custom", "YOLO", "YOLO scaled"]
    OBJECT_CLASSES = [0, 2, 3, 5, 7]

    def __init__(self, project_root):
        self.root = project_root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)

        self.custom = DepthNet(freeze_encoder=False)

        weights_path = self.root / "new_best_depth.pt"
        weights = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )

        self.custom.load_state_dict(weights)
        self.custom.to(self.device)
        self.custom.eval()

        self.yolo = YOLO(str(self.root / "sim" / "yolo26s-depth.pt"))
        self.detector = Detector(str(self.root / "yolo26s.pt"))


        self.pixel_results = {}
        self.object_results = {}
        for model_name in self.MODELS:
            self.pixel_results[model_name] = {}
            self.object_results[model_name] = {}

            for band_name, low, high in self.BANDS:
                self.pixel_results[model_name][band_name] = self.empty_result()
                self.object_results[model_name][band_name] = self.empty_result()

    def empty_result(self):
        return {"count": 0, "absolute_error": 0, "relative_error": 0, "over_1m": 0}



    def predict_custom(self, image):
        image_tensor = to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_depth = self.custom(image_tensor)
            depth = torch.exp(log_depth.clamp(max=7.0))

        return depth[0, 0].cpu().numpy()

    def predict_yolo(self, image_path):
        yolo_device = 0 if self.device == "cuda" else "cpu"

        result = self.yolo.predict(
            source=str(image_path),
            imgsz=768,
            device=yolo_device,
            verbose=False,
        )[0]

        return np.squeeze(result.depth.data.cpu().numpy())
    

    def median_scale(self, prediction, ground_truth):
        valid = ground_truth < 50
        scale = np.median(ground_truth[valid] / np.median(prediction[valid]))
        return prediction * scale


    def box_meadian(self, depth, box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        left = int(x1 + 0.25 * w)
        right = int(x2 - 0.25 * w)
        top = int(y1 + 0.3 * h)
        bottom = int(y2 - 0.1 * h)
        core = depth[top:bottom, left:right]
        if core.size < 10:
            return None
        return float(np.median(core))

    def add_errors(self, results, predicted, ground_truth):
        error = predicted - ground_truth
        results["count"] += error.size
        results["absolute_error"] += np.abs(error).sum()
        results["relative_error"] += (np.abs(error) / ground_truth).sum()
        results["over_1m"] += np.count_nonzero(error > 1.0)


    
    def score_pixels(self,  model_name, predictions, ground_truth):
        for band_name, low, high in self.BANDS:
            mask = (ground_truth > 0) & (ground_truth >= low) & (ground_truth < high)
            if not np.any(mask):
                continue

            result = self.pixel_results[model_name][band_name]
            self.add_errors(result, predictions[mask], ground_truth[mask])


    def score_objects(self, model_name, prediction, ground_truth, boxes):
        for box in boxes:
            true_distance = self.box_meadian(ground_truth, box)
            predicted_distance = self.box_meadian(prediction, box)
            if true_distance is None or predicted_distance is None:
                continue

            for band_name, low, high in self.BANDS:
                if low <= true_distance < high:
                    result  = self.object_results[model_name][band_name]
                    self.add_errors(result, np.array([predicted_distance]), np.array([true_distance]))

    
    def run(self):
        image_paths = []

        for episode in self.EPISODES:
            rgb_folder = self.root / "data" / episode / "rgb"

            for image_path in sorted(rgb_folder.glob("*.png")):
                image_paths.append(image_path)

        if not image_paths:
            raise FileNotFoundError("No test images found")

        for image_path in tqdm(image_paths, desc="Comparing depth"):
            episode_folder = image_path.parent.parent
            depth_folder = episode_folder / "depth"
            depth_path = depth_folder / f"{image_path.stem}.npy"

            ground_truth = np.load(depth_path)

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")


            predictions = {}
            predictions["Custom"] = self.predict_custom(rgb_image)
            predictions["YOLO"] = self.predict_yolo(image_path)
            predictions["YOLO scaled"] = self.median_scale(predictions["YOLO"], ground_truth)

            detection_result = self.detector.predict(str(image_path), classes=self.OBJECT_CLASSES)
            boxes = []
            for detection in self.detector.extract_detections(detection_result):
                boxes.append(detection["bbox"])


            for model_name in self.MODELS:
                self.score_pixels(model_name, predictions[model_name], ground_truth)
                self.score_objects(model_name, predictions[model_name], ground_truth, boxes)


        print(f"\nCompared {len(image_paths)} images\n")
        print("PIXEL LEVEL")
        self.print_table(self.pixel_results)
        print("OBJECT LEVEL")
        self.print_table(self.object_results)
        self.plot_results()

    def get_mae(self, result):
        if result["count"] == 0:
            return float("nan")
        return result["absolute_error"] / result["count"]

    def print_table(self, results):
        for model_name in self.MODELS:
            print(model_name)
            for band_name, low, high in self.BANDS:
                result = results[model_name][band_name]
                count = result["count"]
                if count == 0:
                    print(f"  {band_name}: nothing")
                    continue

                mae = self.get_mae(result)
                abs_rel = result["relative_error"] / count
                over_rate = 100 * result["over_1m"] / count
                print(f"  {band_name}: MAE {mae:.2f} m, AbsRel {abs_rel:.3f}, "
                      f"overestimated by >1 m: {over_rate:.1f}%, n={count}")
            print()

    def plot_results(self):
        band_names = []
        for band_name, low, high in self.BANDS:
            band_names.append(band_name)

        positions = np.arange(len(band_names))
        bar_width = 0.27

        fig, (ax_pixels, ax_objects) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        for i, model_name in enumerate(self.MODELS):
            pixel_mae = []
            object_mae = []
            for band_name in band_names:
                pixel_mae.append(self.get_mae(self.pixel_results[model_name][band_name]))
                object_mae.append(self.get_mae(self.object_results[model_name][band_name]))

            offset = (i - 1) * bar_width
            ax_pixels.bar(positions + offset, pixel_mae, bar_width, label=model_name)
            ax_objects.bar(positions + offset, object_mae, bar_width, label=model_name)

        ax_pixels.set_title("Pixel level")
        ax_objects.set_title("Object level")
        ax_pixels.set_xticks(positions, band_names)
        ax_objects.set_xticks(positions, band_names)
        ax_pixels.set_ylabel("MAE (m)")
        ax_pixels.legend()
        fig.suptitle("CARLA episodes (custom, YOLO zero-shot), lower is better")
        fig.tight_layout()
        fig.savefig(self.root / "docs" / "eval.png")
        plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    evaluator = DepthEvaluator(project_root)
    evaluator.run()