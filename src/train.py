import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_last_block(model, model_name: str, unfreeze_last: int) -> None:
    if model_name == "efficientnet_b0":
        if not hasattr(model, "features"):
            print("WARNING: model.features not found; keeping classifier-only training.")
            return
        # Freeze all feature blocks first.
        for param in model.features.parameters():
            param.requires_grad = False
        num_children = len(model.features)
        blocks_to_unfreeze = max(0, int(unfreeze_last))
        start_idx = max(0, num_children - blocks_to_unfreeze)
        for i in range(start_idx, num_children):
            for param in model.features[i].parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(
            f"EfficientNet unfreeze: requested last={unfreeze_last}, "
            f"using blocks [{start_idx}:{num_children}]"
        )
    else:
        # ResNet fallback: keep prior behavior and ensure classifier remains trainable.
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
    parser.add_argument("--mode", default="eval", choices=["train", "eval", "demo"])
    parser.add_argument("--thresholds", default="0.5,0.6,0.7", help="Comma-separated positive class thresholds.")
    parser.add_argument(
        "--unfreeze-last",
        type=int,
        default=0,
        help="Number of last feature blocks to unfreeze during stage-2 (0 = keep existing stage-2 default behavior)",
    )
    parser.add_argument("--load-from", default="", help="Optional checkpoint path to initialize model weights")
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
    if args.load_from:
        checkpoint_path = Path(args.load_from)
        if not checkpoint_path.exists():
            raise SystemExit(f"Missing checkpoint to load: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise SystemExit("Checkpoint missing model weights (state_dict/model_state_dict)")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model weights from: {checkpoint_path}")

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

    if args.unfreeze_last > 0 and args.lr > 1e-4:
        print("Warning: High LR detected with unfreeze - scaling down to 1e-5")
        args.lr = 1e-5

    if args.model == "efficientnet_b0" and args.unfreeze_last > 0:
        print("=== DEBUG: UNFREEZE AFTER LOADING CHECKPOINT ===")
        print(
            "Before unfreeze - requires_grad status of first feature param: "
            f"{next(model.features.parameters()).requires_grad}"
        )
        print("Applying unfreeze: freezing all features first")
        try:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

            num_blocks = len(model.features)
            print(f"model.features has {num_blocks} children (should be ~9 for B0)")
            start_idx = max(0, num_blocks - args.unfreeze_last)
            print(
                f"Unfreezing from index {start_idx} to {num_blocks - 1} "
                f"(last {args.unfreeze_last} blocks)"
            )
            for i in range(start_idx, num_blocks):
                block = model.features[i]
                for param in model.features[i].parameters():
                    param.requires_grad = True
                trainable_in_block = sum(1 for p in block.parameters() if p.requires_grad)
                print(f"  Block {i}: {trainable_in_block} params set to requires_grad=True")

            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            total = sum(1 for p in model.parameters())
            trainable_params_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"=== AFTER UNFROZE: Trainable layers: {trainable}/{total} | "
                f"Trainable params: {trainable_params_numel:,} "
                f"({trainable_params_numel / max(1, sum(p.numel() for p in model.parameters())) * 100:.2f}%) ==="
            )
        except Exception as ex:
            print(f"WARNING: EfficientNet unfreeze failed ({ex}); using head-only mode")
            freeze_backbone(model, args.model)
            print(f"Head-only stage trainable parameters: {count_trainable_params(model):,}")
    else:
        freeze_backbone(model, args.model)
        print(f"Head-only stage trainable parameters: {count_trainable_params(model):,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"Optimizer created with {len(optimizer.param_groups[0]['params'])} parameters "
        f"(trainable tensors in first group)"
    )
    print(f"Initial LR set to {args.lr:.2e}")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
    )
    print(f"Starting training with LR={args.lr}, unfreeze-last={args.unfreeze_last}")

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

        lr_value = optimizer.param_groups[0]["lr"]
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

        scheduler.step(val_loss)

        if stopper.step(val_loss):
            print("Early stopping triggered.")
            return True
        return False

    for epoch in range(1, args.epochs + 1):
        if run_epoch(epoch, args.epochs):
            break
    print(f"Best model saved to {output_path} (best_threshold={best_threshold:.2f})")


if __name__ == "__main__":
    main()
