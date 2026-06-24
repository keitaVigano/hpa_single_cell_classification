from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os


class HPADataset(Dataset):
    def __init__(self, image_ids, label_map, path_data, path_masks, img_size=224):
        self.label_map = label_map
        self.path_data = path_data
        self.path_masks = path_masks
        self.img_size = img_size

        self.samples = []
        for image_id in image_ids:
            if image_id not in label_map:
                continue
            mask = self.load_img_channel(image_id, path_masks)
            if mask is None:
                continue
            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids != 0]
            for cell_id in cell_ids:
                self.samples.append((image_id, cell_id))

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

    def __getitem__(self, idx):
        image_id, cell_id = self.samples[idx]

        img = self.load_img(image_id, self.path_data)
        mask = self.load_img_channel(image_id, self.path_masks)

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
        mask_path: str = f"{path}/{image_id}.png"
        if not os.path.exists(mask_path):
            return None
        return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
