import pandas as pd
import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CsvImageDataset
from model import build_model
from transforms import get_val_transforms
from utils import compute_metrics


def parse_thresholds(raw: str) -> list[float]:
    if not raw or not raw.strip():
        return [round(0.3 + (0.05 * i), 2) for i in range(11)]
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
    parser = argparse.ArgumentParser(description="Evaluate model on test set.")
    parser.add_argument("--csv", default="data/test_labels.csv", help="Path to test_labels.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mode", default="eval", choices=["eval", "demo"])
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated positive class thresholds. Default sweep: 0.3 to 0.8 (step 0.05).",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to export prediction error analysis.")
    parser.add_argument(
        "--copy-misclassified-images",
        action="store_true",
        help="Copy false-positive/false-negative images for manual visual review.",
    )
    args = parser.parse_args()
    thresholds = parse_thresholds(args.thresholds)

    csv_path = Path(args.csv)
    checkpoint_path = Path(args.checkpoint)
    if not csv_path.exists() and args.mode == "eval":
        raise SystemExit(f"Missing CSV: {csv_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    if args.mode == "demo":
        train_path = Path("data/train_labels.csv")
        test_path = Path("data/test_labels.csv")
        if not train_path.exists():
            raise SystemExit(f"Missing CSV: {train_path}")
        df = pd.read_csv(train_path)
        if test_path.exists():
            df = pd.concat([df, pd.read_csv(test_path)], ignore_index=True)
        demo_path = Path("data/demo_labels.csv")
        df.to_csv(demo_path, index=False)
        dataset = CsvImageDataset(demo_path, transform=get_val_transforms())
    else:
        dataset = CsvImageDataset(csv_path, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", "resnet18")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    targets = []
    probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            prob_pos = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            targets.extend(labels.numpy().tolist())
            probs.extend(prob_pos.tolist())

    metric_by_threshold = [
        compute_metrics(targets, probs=probs, threshold=threshold) for threshold in thresholds
    ]
    metrics = max(metric_by_threshold, key=lambda metric: metric.f1)
    best_threshold = metrics.threshold
    final_preds = (pd.Series(probs).to_numpy() >= best_threshold).astype(int)
    print("Threshold sweep (positive class probability):")
    for metric in metric_by_threshold:
        print(
            f"  thr={metric.threshold:.2f} "
            f"precision={metric.precision:.4f} "
            f"recall={metric.recall:.4f} "
            f"f1={metric.f1:.4f}"
        )

    # Export per-sample predictions and misclassifications for error analysis.
    label_name = {0: "URTICARIA", 1: "ANGIOEDEMA"}
    eval_df = dataset.data.reset_index(drop=True).copy()
    if "abs_path" in eval_df.columns:
        image_paths = eval_df["abs_path"].astype(str)
    elif "image_path" in eval_df.columns:
        image_paths = eval_df["image_path"].astype(str)
    else:
        image_paths = pd.Series([""] * len(eval_df), dtype=str)

    all_predictions_df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [label_name.get(int(value), str(value)) for value in targets],
        "predicted_label": [label_name.get(int(value), str(value)) for value in final_preds],
        "positive_probability": probs,
        "threshold_used": best_threshold,
    })
    all_predictions_df["error_type"] = all_predictions_df.apply(
        lambda row: (
            "TP" if row["true_label"] == "ANGIOEDEMA" and row["predicted_label"] == "ANGIOEDEMA"
            else "TN" if row["true_label"] == "URTICARIA" and row["predicted_label"] == "URTICARIA"
            else "FP" if row["true_label"] == "URTICARIA" and row["predicted_label"] == "ANGIOEDEMA"
            else "FN"
        ),
        axis=1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_csv = output_dir / "all_predictions.csv"
    false_positives_csv = output_dir / "false_positives.csv"
    false_negatives_csv = output_dir / "false_negatives.csv"
    all_predictions_df.to_csv(all_predictions_csv, index=False)
    all_predictions_df[all_predictions_df["error_type"] == "FP"].to_csv(false_positives_csv, index=False)
    all_predictions_df[all_predictions_df["error_type"] == "FN"].to_csv(false_negatives_csv, index=False)
    print(f"Exported all predictions: {all_predictions_csv}")
    print(f"Exported false positives: {false_positives_csv}")
    print(f"Exported false negatives: {false_negatives_csv}")

    if args.copy_misclassified_images:
        fp_dir = output_dir / "false_positives_images"
        fn_dir = output_dir / "false_negatives_images"
        fp_dir.mkdir(parents=True, exist_ok=True)
        fn_dir.mkdir(parents=True, exist_ok=True)

        for _, row in all_predictions_df[all_predictions_df["error_type"] == "FP"].iterrows():
            source = Path(str(row["image_path"]))
            if source.exists():
                shutil.copy2(source, fp_dir / source.name)
        for _, row in all_predictions_df[all_predictions_df["error_type"] == "FN"].iterrows():
            source = Path(str(row["image_path"]))
            if source.exists():
                shutil.copy2(source, fn_dir / source.name)
        print(f"Copied FP images to: {fp_dir}")
        print(f"Copied FN images to: {fn_dir}")

    tn, fp, fn, tp = metrics.confusion_matrix
    if args.mode == "demo":
        print("Demo evaluation on seen data (sanity check only).")
        print(f"Best threshold by F1: {metrics.threshold:.2f}")
        print(f"AUC: {metrics.auc:.4f}")
        print("Confusion Matrix:")
        print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    else:
        print(f"Best threshold by F1: {metrics.threshold:.2f}")
        print("Accuracy:", metrics.accuracy)
        print("Precision:", metrics.precision)
        print("Recall:", metrics.recall)
        print("F1:", metrics.f1)
        print("AUC:", metrics.auc)
        print("Per-class metrics:")
        for class_name, class_metrics in metrics.per_class.items():
            print(
                f"{class_name}: "
                f"precision={class_metrics['precision']:.4f} "
                f"recall={class_metrics['recall']:.4f} "
                f"f1={class_metrics['f1']:.4f}"
            )
        print("Confusion Matrix:")
        print(f"TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
