"""
3-stage fine-tuning script for URTICARIA vs ANGIOEDEMA classifier.

Stage 1  – Head only        (15 epochs, lr=1e-3)
Stage 2  – Unfreeze last 3  (25 epochs, lr=2e-4, layer-wise decay)
Stage 3  – Full fine-tune   (20 epochs, lr=3e-5, cosine annealing)

Key improvements over the original train.py:
  * Much stronger augmentation to break watermark / background shortcuts
  * RandomErasing to prevent the model learning the ©Dermnet watermark
  * CosineAnnealingLR for smooth LR decay (better than plateau scheduler)
  * Gradient clipping to stabilise full-model fine-tuning
  * Label smoothing 0.1 (up from 0.05)
  * Layer-wise LR: newly unfrozen blocks use lower LR than the head
  * Mixup augmentation in stage 2/3 for better generalisation
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CsvImageDataset
from model import build_model
from utils import EarlyStopping, compute_metrics, set_seed


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def strong_train_transforms(image_size: int = 224):
    """
    Heavy augmentation to prevent shortcut learning on small medical datasets.

    Key additions vs original:
    - Wider crop scale (0.5-1.0)   → forces multi-scale lesion features
    - Stronger rotation (±30°)     → removes orientation shortcuts
    - Translation + shear          → further breaks spatial shortcuts
    - Strong colour jitter          → breaks lighting/watermark colour cues
    - RandomGrayscale              → reduces colour-only cues
    - RandomErasing (p=0.5)        → randomly occludes the ©Dermnet watermark
    - GaussianBlur (random apply)  → texure robustness
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.06),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
    ])


def val_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3):
    """Apply mixup to a batch. Returns mixed images and (labels_a, labels_b, lam)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model, model_name: str):
    freeze_all(model)
    if model_name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.fc.parameters():
            p.requires_grad = True


def unfreeze_last_n(model, model_name: str, n: int):
    """Unfreeze the last n feature blocks + classifier head."""
    freeze_all(model)
    if model_name == "efficientnet_b0":
        blocks = list(model.features.children())
        for block in blocks[max(0, len(blocks) - n):]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True


def unfreeze_full(model):
    for p in model.parameters():
        p.requires_grad = True


def make_param_groups(model, model_name: str, head_lr: float, body_lr: float):
    """Layer-wise LR: head gets head_lr, unfrozen body gets body_lr."""
    if model_name == "efficientnet_b0":
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        body_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        return [
            {"params": head_params, "lr": head_lr},
            {"params": body_params, "lr": body_lr},
        ]
    else:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": head_lr}]


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    use_mixup: bool = False,
    grad_clip: float = 1.0,
) -> tuple[float, list, list]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_targets: list = []
    all_probs: list = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)

            if training and use_mixup and np.random.rand() < 0.5:
                images, la, lb, lam = mixup_batch(images, labels)
                logits = model(images)
                loss = mixup_loss(criterion, logits, la, lb, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())

    return total_loss / max(len(loader.dataset), 1), all_targets, all_probs


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(
    stage_name: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler,
    criterion,
    epochs: int,
    device: torch.device,
    output_path: Path,
    model_name: str,
    best_f1: float,
    best_threshold: float,
    use_mixup: bool = False,
    early_patience: int = 8,
):
    stopper = EarlyStopping(patience=early_patience, min_delta=1e-4)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for epoch in range(1, epochs + 1):
        train_loss, t_tgt, t_prb = run_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup
        )
        val_loss, v_tgt, v_prb = run_epoch(
            model, val_loader, criterion, None, device
        )

        thr_metrics = [
            compute_metrics(v_tgt, probs=np.array(v_prb), threshold=thr)
            for thr in thresholds
        ]
        best_val = max(thr_metrics, key=lambda m: m.f1)
        t_met = compute_metrics(t_tgt, probs=np.array(t_prb), threshold=0.5)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{stage_name}] ep {epoch:02d}/{epochs} "
            f"lr={lr_now:.1e} "
            f"trn_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"trn_acc={t_met.accuracy:.3f} val_acc={best_val.accuracy:.3f} "
            f"val_f1={best_val.f1:.3f} val_auc={best_val.auc:.3f} "
            f"thr={best_val.threshold:.2f}"
        )

        if best_val.f1 > best_f1:
            best_f1 = best_val.f1
            best_threshold = best_val.threshold
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "best_threshold": best_threshold,
                },
                output_path,
            )
            print(f"  ✓ Saved best (f1={best_f1:.4f}, thr={best_threshold:.2f})")

        scheduler.step()

        if stopper.step(val_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    return best_f1, best_threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-stage fine-tuning for urticaria classifier")
    parser.add_argument("--csv", default="data/train_labels.csv")
    parser.add_argument("--model", default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/best.pt")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    # Per-stage epoch counts
    parser.add_argument("--s1-epochs", type=int, default=15, help="Stage 1: head-only epochs")
    parser.add_argument("--s2-epochs", type=int, default=25, help="Stage 2: unfreeze last 3 blocks")
    parser.add_argument("--s3-epochs", type=int, default=20, help="Stage 3: full fine-tune")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(len(idx) * (1 - args.val_ratio))
    train_df = df.iloc[idx[:split]].reset_index(drop=True)
    val_df   = df.iloc[idx[split:]].reset_index(drop=True)

    train_labels_arr = np.array(
        [1 if str(l).upper() == "ANGIOEDEMA" else 0 for l in train_df["label"]], dtype=np.int64
    )
    class_counts = np.bincount(train_labels_arr, minlength=2)
    total = class_counts.sum()
    class_weights = torch.tensor(
        total / (2 * (class_counts + 1e-6)), dtype=torch.float32
    ).to(device)
    print(f"Class counts (train) URTICARIA={class_counts[0]} ANGIOEDEMA={class_counts[1]}")
    print(f"Class weights: {class_weights.tolist()}")

    train_csv = csv_path.parent / "train_split.csv"
    val_csv   = csv_path.parent / "val_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = CsvImageDataset(train_csv, transform=strong_train_transforms())
    val_ds   = CsvImageDataset(val_csv,   transform=val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- Model ----
    model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    best_f1, best_thr = -1.0, 0.5

    # ==================================================================
    # STAGE 1 — Head only
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Head-only training")
    print("=" * 60)
    unfreeze_head(model, args.model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4,
    )
    sch1 = CosineAnnealingLR(opt1, T_max=args.s1_epochs, eta_min=1e-5)
    best_f1, best_thr = run_stage(
        "S1-head", model, train_loader, val_loader,
        opt1, sch1, criterion, args.s1_epochs,
        device, output_path, args.model,
        best_f1, best_thr, use_mixup=False, early_patience=6,
    )

    # Load best stage-1 weights before unfreezing
    ckpt = torch.load(output_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # ==================================================================
    # STAGE 2 — Unfreeze last 3 feature blocks
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Unfreeze last 3 feature blocks")
    print("=" * 60)
    unfreeze_last_n(model, args.model, n=3)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    param_groups = make_param_groups(model, args.model, head_lr=2e-4, body_lr=5e-5)
    opt2 = AdamW(param_groups, weight_decay=1e-4)
    sch2 = CosineAnnealingLR(opt2, T_max=args.s2_epochs, eta_min=1e-6)
    best_f1, best_thr = run_stage(
        "S2-unfreeze3", model, train_loader, val_loader,
        opt2, sch2, criterion, args.s2_epochs,
        device, output_path, args.model,
        best_f1, best_thr, use_mixup=True, early_patience=8,
    )

    # Load best stage-2 weights
    ckpt = torch.load(output_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # ==================================================================
    # STAGE 3 — Full fine-tune
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Full fine-tune (all layers)")
    print("=" * 60)
    unfreeze_full(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    if args.model == "efficientnet_b0":
        # Layer-wise LR: head → 3e-5, rest → 1e-5
        param_groups = make_param_groups(model, args.model, head_lr=3e-5, body_lr=1e-5)
    else:
        param_groups = [{"params": model.parameters(), "lr": 1e-5}]

    opt3 = AdamW(param_groups, weight_decay=1e-4)
    sch3 = CosineAnnealingLR(opt3, T_max=args.s3_epochs, eta_min=1e-7)
    best_f1, best_thr = run_stage(
        "S3-fulltune", model, train_loader, val_loader,
        opt3, sch3, criterion, args.s3_epochs,
        device, output_path, args.model,
        best_f1, best_thr, use_mixup=True, early_patience=10,
    )

    print("\n" + "=" * 60)
    print(f"Training complete.  Best val F1={best_f1:.4f}  threshold={best_thr:.2f}")
    print(f"Best checkpoint: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
