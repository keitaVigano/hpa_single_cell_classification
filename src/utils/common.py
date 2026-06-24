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