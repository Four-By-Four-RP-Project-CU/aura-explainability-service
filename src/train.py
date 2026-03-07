import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import CsvImageDataset
from model import build_model
from transforms import get_train_transforms, get_val_transforms
from utils import EarlyStopping, compute_metrics, set_seed


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def freeze_backbone(model, model_name: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_last_block(model, model_name: str) -> None:
    if model_name == "efficientnet_b0":
        for param in model.features[-2].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True


def parse_thresholds(raw: str) -> list[float]:
    thresholds = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if 0.0 < value < 1.0:
            thresholds.append(value)
    if not thresholds:
        return [0.5]
    return sorted(set(thresholds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train URTICARIA vs ANGIOEDEMA classifier.")
    parser.add_argument("--csv", default="data/train_labels.csv", help="Path to train_labels.csv")
    parser.add_argument("--model", default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/best.pt")
    parser.add_argument("--mode", default="eval", choices=["eval", "demo"])
    parser.add_argument("--thresholds", default="0.5,0.6,0.7", help="Comma-separated positive class thresholds.")
    args = parser.parse_args()

    set_seed(args.seed)
    thresholds = parse_thresholds(args.thresholds)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    csv_labels = np.array([1 if str(label).upper() == "ANGIOEDEMA" else 0 for label in df["label"]], dtype=np.int64)
    csv_class_counts = np.bincount(csv_labels, minlength=2)
    print(f"Training CSV class counts (0=URTICARIA, 1=ANGIOEDEMA): {csv_class_counts.tolist()}")

    if args.mode == "demo":
        test_path = csv_path.parent / "test_labels.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            df = pd.concat([df, test_df], ignore_index=True)
        train_df, val_df = split_train_val(df, val_ratio=0.1, seed=args.seed)
    else:
        train_df, val_df = split_train_val(df, val_ratio=0.2, seed=args.seed)

    train_df_path = csv_path.parent / "train_split.csv"
    val_df_path = csv_path.parent / "val_split.csv"
    train_df.to_csv(train_df_path, index=False)
    val_df.to_csv(val_df_path, index=False)

    train_dataset = CsvImageDataset(train_df_path, transform=get_train_transforms())
    val_dataset = CsvImageDataset(val_df_path, transform=get_val_transforms())

    train_labels = np.array([1 if str(label).upper() == "ANGIOEDEMA" else 0 for label in train_df["label"]], dtype=np.int64)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=2).to(device)

    class_counts = np.bincount(train_labels, minlength=2)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * (class_counts + 1e-6))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class counts (0=URTICARIA, 1=ANGIOEDEMA): {class_counts.tolist()}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")
    print(f"Threshold candidates: {thresholds}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    stopper = EarlyStopping(patience=5, min_delta=1e-4)

    best_f1 = -1.0
    best_threshold = 0.5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_epochs = min(8, max(5, args.epochs // 5))
    stage2_epochs = max(1, args.epochs - stage1_epochs)

    freeze_backbone(model, args.model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs)

    def run_epoch(epoch_index, total_epochs):
        nonlocal best_f1, best_threshold
        model.train()
        train_loss = 0.0
        train_targets = []
        train_probs = []
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch_index}/{total_epochs} - train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_targets.extend(labels.detach().cpu().numpy().tolist())
            probs = torch.softmax(logits, dim=1)[:, 1]
            train_probs.extend(probs.detach().cpu().numpy().tolist())

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch_index}/{total_epochs} - val"):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_targets.extend(labels.cpu().numpy().tolist())
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy().tolist())

        val_loss /= len(val_loader.dataset)

        train_metrics = compute_metrics(train_targets, probs=np.array(train_probs), threshold=0.5)
        threshold_metrics = [
            compute_metrics(val_targets, probs=np.array(val_probs), threshold=thr) for thr in thresholds
        ]
        val_metrics = max(threshold_metrics, key=lambda metric: metric.f1)

        lr_value = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch_index}: lr={lr_value:.2e} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} val_acc={val_metrics.accuracy:.3f} "
            f"train_f1={train_metrics.f1:.3f} val_f1={val_metrics.f1:.3f} "
            f"train_auc={train_metrics.auc:.3f} val_auc={val_metrics.auc:.3f} "
            f"thr={val_metrics.threshold:.2f}"
        )

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_threshold = val_metrics.threshold
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "best_threshold": best_threshold,
                },
                output_path,
            )

        scheduler.step()

        if stopper.step(val_loss):
            print("Early stopping triggered.")
            return True
        return False

    for epoch in range(1, stage1_epochs + 1):
        if run_epoch(epoch, args.epochs):
            print(f"Best model saved to {output_path}")
            return

    unfreeze_last_block(model, args.model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)

    for epoch in range(stage1_epochs + 1, args.epochs + 1):
        if run_epoch(epoch, args.epochs):
            break
    print(f"Best model saved to {output_path} (best_threshold={best_threshold:.2f})")


if __name__ == "__main__":
    main()
