import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthDataset(Dataset):
    def __init__(self, root, episodes, max_depth=50.0):
        self.root = Path(root)
        self.max_depth = max_depth
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.samples = []
        for ep in episodes:
            rgb_dir = self.root / ep / "rgb"
            depth_dir = self.root / ep / "depth"
            for rgb_path in sorted(rgb_dir.glob("*.png")):
                depth_path = depth_dir / f"{rgb_path.stem}.npy"
                if not depth_path.exists():
                    raise FileNotFoundError(f"no matching depth file for {rgb_path}")
                self.samples.append((rgb_path, depth_path))

        if not self.samples:
            raise RuntimeError(f"no samples found for episodes={episodes} under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        image = Image.open(rgb_path).convert("RGB")
        image = T.functional.to_tensor(image)
        image = self.normalize(image)

        depth = np.load(depth_path).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)

        mask = depth < self.max_depth
        depth = torch.clamp(depth, max=self.max_depth)

        return image, depth, mask


def list_episodes(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir())




class DataAug(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data_list[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path)

        rgb = TF.to_tensor(rgb)
        depth = torch.from_numpy(depth).unsqueeze(0)

        if random.random() > 0.5:
            rgb = self.color_jitter(rgb)

        i, j, h, w = T.RandomCrop.get_params(rgb, output_size=(320, 600))
        rgb = TF.crop(rgb, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        return rgb, depth

    def __len__(self):
      return len(self.data_list)