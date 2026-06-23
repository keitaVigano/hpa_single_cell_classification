import numpy as np
import torch
from sklearn.metrics import f1_score


def multilabel_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    logits: (N, num_classes) raw output del modello (pre-sigmoid)
    targets: (N, num_classes) multi-hot ground truth
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > threshold).astype(np.int32)
    targets_np = targets.detach().cpu().numpy().astype(np.int32)

    return {
        "f1_macro": f1_score(targets_np, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(targets_np, preds, average="micro", zero_division=0),
    }
