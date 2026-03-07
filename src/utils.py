import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    confusion_matrix: Tuple[int, int, int, int]
    per_class: Dict[str, Dict[str, float]]
    threshold: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc_binary(targets: np.ndarray, scores: np.ndarray) -> float:
    pos = int((targets == 1).sum())
    neg = int((targets == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[targets == 1].sum()
    auc = (sum_ranks_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def compute_metrics(
    targets,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Metrics:
    targets = np.array(targets)
    if preds is None:
        if probs is None:
            raise ValueError("Either preds or probs must be provided.")
        preds = (np.array(probs) >= threshold).astype(int)
    else:
        preds = np.array(preds)

    if probs is None:
        probs = preds.astype(np.float64)
    else:
        probs = np.array(probs, dtype=np.float64)

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    total = tp + tn + fp + fn

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Class 1 (ANGIOEDEMA)
    c1_precision = precision
    c1_recall = recall
    c1_f1 = f1
    # Class 0 (URTICARIA)
    c0_precision = _safe_div(tn, tn + fn)
    c0_recall = _safe_div(tn, tn + fp)
    c0_f1 = _safe_div(2 * c0_precision * c0_recall, c0_precision + c0_recall)

    auc = _roc_auc_binary(targets, probs)
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        confusion_matrix=(tn, fp, fn, tp),
        per_class={
            "URTICARIA": {
                "precision": c0_precision,
                "recall": c0_recall,
                "f1": c0_f1,
            },
            "ANGIOEDEMA": {
                "precision": c1_precision,
                "recall": c1_recall,
                "f1": c1_f1,
            },
        },
        threshold=threshold,
    )


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
