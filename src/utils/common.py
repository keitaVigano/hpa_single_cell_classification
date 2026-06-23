import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.model import HPAResNet
from sklearn.model_selection import train_test_split
from src.dataset import HPADataset
import numpy as np
import pandas as pd
import torch.nn as nn
import wandb

def load_img(image_id: str, path: str) -> np.ndarray:
    red = load_img_channel(f"{image_id}_red", path)
    green = load_img_channel(f"{image_id}_green", path)
    blue = load_img_channel(f"{image_id}_blue", path)
    yellow = load_img_channel(f"{image_id}_yellow", path)
    if red is None or blue is None or yellow is None or green is None:
        raise FileNotFoundError(f"Missing one or more channel files for image_id={image_id} in {path}")
    return np.stack([red, green, blue, yellow], axis=-1)  # (H, W, 4)


def norm(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    return (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)


def load_img_channel(image_id: str, path: str) -> np.ndarray | None:
    mask_path: str = f"{path}/{image_id}.png"
    if not os.path.exists(mask_path):
        return None
    return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def plot_img(image_ids: set[str], path: str) -> None:
    sample_ids: list[str] = list(image_ids)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, image_id in enumerate(sample_ids):
        img = load_img(image_id, path)
        rgb = np.stack([norm(img[..., 0]), norm(img[..., 1]), norm(img[..., 2])], axis=-1)
        axes[i].imshow(rgb)
        axes[i].axis("off")
        axes[i].set_title(image_id[:8], fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_img_with_mask(image_ids: set[str], path_data: str, path_mask: str) -> None:

    sample_ids = list(image_ids)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, image_id in enumerate(sample_ids):
        img = load_img(image_id, path_data)
        rgb = np.stack([norm(img[..., 0]), norm(img[..., 1]), norm(img[..., 2])], axis=-1)

        mask = load_img_channel(image_id, path_mask)

        axes[i].imshow(rgb)

        if mask is not None:
            axes[i].imshow(mask, cmap='tab20', alpha=0.4, interpolation='nearest')

        axes[i].axis('off')
        axes[i].set_title(image_id[:8], fontsize=8)

    plt.tight_layout()
    plt.show()


def label_to_vector(label_str: str) -> np.ndarray:
    vector = np.zeros(19, dtype=np.float32)
    for idx in label_str.split("|"):
        vector[int(idx)] = 1.0
    return vector

def get_model(model: str, num_classes: int = 19) -> nn.Module:
    if model == "resnet":
        return HPAResNet(num_classes=num_classes)
    else:
        raise ValueError("The possible models are: resnet")


def build_label_map(csv_path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    return {row.ID: label_to_vector(row.Label) for row in df.itertuples()}


def get_train_val_datasets(
    csv_path: str,
    path_data: str,
    path_masks: str,
    val_split: float = 0.2,
    img_size: int = 224,
    random_state: int = 42,
) -> tuple[HPADataset, HPADataset]:
    label_map = build_label_map(csv_path)
    all_image_ids = list(label_map.keys())

    # split a livello di image_id, NON di cella → evita leakage
    train_ids, val_ids = train_test_split(
        all_image_ids,
        test_size=val_split,
        random_state=random_state,
    )

    train_dataset = HPADataset(
        image_ids=train_ids,
        label_map=label_map,
        path_data=path_data,
        path_masks=path_masks,
        img_size=img_size,
    )
    val_dataset = HPADataset(
        image_ids=val_ids,
        label_map=label_map,
        path_data=path_data,
        path_masks=path_masks,
        img_size=img_size,
    )
    return train_dataset, val_dataset


def init_wandb_logger(
    model: nn.Module,
    project: str,
    run_name: str | None = None,
    config: dict | None = None,
    watch_model: bool = True,
) -> wandb.sdk.wandb_run.Run:
    run = wandb.init(
        project=project,
        name=run_name,
        config=config or {},
    )
    if watch_model:
        wandb.watch(model, log="gradients", log_freq=100)
    return run