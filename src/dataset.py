from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_INDEX = {
    "URTICARIA": 0,
    "ANGIOEDEMA": 1,
}


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, row) -> Path:
        abs_path = str(row.get("abs_path", "")).strip()
        if abs_path:
            return Path(abs_path)
        image_path = str(row.get("image_path", "")).strip()
        if image_path:
            return Path(image_path)
        raise ValueError("Row does not contain abs_path or image_path")

    def _resolve_label(self, row) -> int:
        raw = str(row.get("label", "")).strip().upper()
        if raw in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[raw]
        return 1 if raw == "1" else 0

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image_path = self._resolve_image_path(row)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self._resolve_label(row), dtype=torch.long)
        return image, label, str(image_path)
