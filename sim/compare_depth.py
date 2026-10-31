from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from ultralytics import YOLO

from depth.dataset import IMAGENET_MEAN, IMAGENET_STD
from depth.model import DepthNet


class DepthEvaluator:
    EPISODES = [
        "Town01_ClearNoon_s101",
        "Town05_WetCloudyNoon_s105",
    ]

    BANDS = [
        ("0-4 m", 0, 4),
        ("4-8 m", 4, 8),
        ("8-10 m", 8, 10),
    ]

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

        self.yolo = YOLO("yolo26s-depth.pt")

        self.results = {}
        for model_name in ("Custom", "YOLO"):
            self.results[model_name] = {}

            for band_name, low, high in self.BANDS:
                self.results[model_name][band_name] = {
                    "count": 0,
                    "absolute_error": 0.0,
                    "over_1m": 0,
                }

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

        return result.depth.data.cpu().numpy()

    def score(self, model_name, prediction, ground_truth):
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"{model_name}: prediction and ground truth sizes differ"
            )

        for band_name, low, high in self.BANDS:
            mask = (
                np.isfinite(ground_truth)
                & (ground_truth > 0)
                & (ground_truth >= low)
                & (ground_truth < high)
            )

            if not np.any(mask):
                continue

            predicted_values = prediction[mask]

            if not np.all(
                np.isfinite(predicted_values) & (predicted_values > 0)
            ):
                raise ValueError(f"{model_name} returned invalid depth")

            error = predicted_values - ground_truth[mask]
            result = self.results[model_name][band_name]

            result["count"] += error.size
            result["absolute_error"] += np.abs(error).sum(
                dtype=np.float64
            )
            result["over_1m"] += np.count_nonzero(error > 1.0)

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

            ground_truth = np.load(depth_path, allow_pickle=False)

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")

            custom_depth = self.predict_custom(rgb_image)
            yolo_depth = self.predict_yolo(image_path)

            self.score("Custom", custom_depth, ground_truth)
            self.score("YOLO", yolo_depth, ground_truth)

        print(f"\nCompared {len(image_paths)} images\n")
        self.show_results()
        self.plot_results()

    def get_mae(self, model_name, band_name):
        result = self.results[model_name][band_name]
        count = result["count"]

        if count == 0:
            return float("nan")

        return result["absolute_error"] / count

    def show_results(self):
        for model_name in ("Custom", "YOLO"):
            print(model_name)

            for band_name, low, high in self.BANDS:
                result = self.results[model_name][band_name]
                count = result["count"]

                if count == 0:
                    print(f"  {band_name}: no pixels")
                    continue

                mae = self.get_mae(model_name, band_name)
                over_rate = 100 * result["over_1m"] / count

                print(
                    f"  {band_name}: MAE {mae:.2f} m, "
                    f"overestimated by >1 m: {over_rate:.1f}%"
                )

            print()

    def plot_results(self):
        band_names = []
        custom_mae = []
        yolo_mae = []

        for band_name, low, high in self.BANDS:
            band_names.append(band_name)
            custom_mae.append(self.get_mae("Custom", band_name))
            yolo_mae.append(self.get_mae("YOLO", band_name))

        positions = np.arange(len(band_names))
        bar_width = 0.35

        plt.figure(figsize=(8, 4))
        plt.bar(
            positions - bar_width / 2,
            custom_mae,
            bar_width,
            label="Custom",
        )
        plt.bar(
            positions + bar_width / 2,
            yolo_mae,
            bar_width,
            label="YOLO depth",
        )

        plt.xticks(positions, band_names)
        plt.ylabel("MAE (metres)")
        plt.title("Depth error within 10 m — lower is better")
        plt.legend()
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    evaluator = DepthEvaluator(project_root)
    evaluator.run()