from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

class HPADataset(Dataset):
    def __init__(self, image_ids, label_map, path_data, path_masks, img_size=224):
        self.label_map = label_map
        self.path_data = path_data
        self.path_masks = path_masks
        self.img_size = img_size
        self._img_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}

        cell_masks_dir = os.path.join(path_masks, "cell_masks_v5", "train")
        nuc_masks_dir = os.path.join(path_masks, "nuc_masks_v5", "train")

        self.samples = []
        for image_id in tqdm(image_ids, desc=f"Building samples ({len(image_ids)} images)"):
            if image_id not in label_map:
                continue

            mask = None
            mask_dir = None

            cell_mask_path = os.path.join(cell_masks_dir, f"{image_id}.png")
            nuc_mask_path = os.path.join(nuc_masks_dir, f"{image_id}.png")

            if os.path.exists(cell_mask_path):
                mask_dir = cell_masks_dir
            elif os.path.exists(nuc_mask_path):
                mask_dir = nuc_masks_dir
            else:
                continue

            mask = self.load_img_channel(image_id, mask_dir)
            if mask is None:
                continue

            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids != 0]
            for cell_id in cell_ids:
                self.samples.append((image_id, cell_id, mask_dir))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_crops(img: np.ndarray, cell_mask: np.ndarray, cell_id: int, img_size: int) -> np.ndarray:
        rows, cols = np.where(cell_mask == cell_id)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        crop = img[y_min:y_max + 1, x_min:x_max + 1, :]
        crop = cv2.resize(crop, (img_size, img_size))
        return crop

    def _get_image(self, image_id: str) -> np.ndarray:
        if image_id not in self._img_cache:
            self._img_cache[image_id] = self.load_img(image_id, self.path_data)
        return self._img_cache[image_id]

    def _get_mask(self, image_id: str, mask_dir: str) -> np.ndarray:
        if image_id not in self._mask_cache:
            self._mask_cache[image_id] = self.load_img_channel(image_id, mask_dir)
        return self._mask_cache[image_id]

    def __getitem__(self, idx):
        image_id, cell_id, mask_dir = self.samples[idx]

        img = self._get_image(image_id)
        mask = self._get_mask(image_id, mask_dir)

        crop = HPADataset.get_crops(img, mask, cell_id, self.img_size)
        crop = crop.astype(np.float32) / 65535.0
        crop = torch.from_numpy(crop).permute(2, 0, 1)
        label = torch.from_numpy(self.label_map[image_id])

        return crop, label

    @staticmethod
    def load_img(image_id: str, path: str) -> np.ndarray:
        red = HPADataset.load_img_channel(f"{image_id}_red", path)
        green = HPADataset.load_img_channel(f"{image_id}_green", path)
        blue = HPADataset.load_img_channel(f"{image_id}_blue", path)
        yellow = HPADataset.load_img_channel(f"{image_id}_yellow", path)
        if red is None or blue is None or yellow is None or green is None:
            raise FileNotFoundError(f"Missing one or more channel files for image_id={image_id} in {path}")
        return np.stack([red, green, blue, yellow], axis=-1)  # (H, W, 4)

    @staticmethod
    def norm(img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        return (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

    @staticmethod
    def load_img_channel(image_id: str, path: str) -> np.ndarray | None:
        mask_path = os.path.join(path, f"{image_id}.png")
        if not os.path.exists(mask_path):
            return None
        return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)