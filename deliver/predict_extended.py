"""Optional flower names/confidence, while preserving the original predict.py.

Usage (from the repository root):
  python deliver/predict_extended.py flower_model_fp32.pt flower_data/test/1/image_06743.jpg
  python deliver/predict_extended.py flower_model_fp32.pt flower_data/test/1/image_06743.jpg --names results/class_to_name_mapping.csv

The original deliver/predict.py is still the competition-safe minimal interface.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from predict import FlowerClassifier


def load_flower_names(path: str | Path) -> dict[str, str]:
    """Read team CSV (class_id,flower_name) or Oxford-style cat_to_name.json.

    Do not assume arbitrary external mappings match the dataset's IDs: verify them.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Flower name mapping not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON mapping must be a dictionary: class ID -> name")
        pairs = [(str(k), str(v)) for k, v in data.items()]
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"class_id", "flower_name"}.issubset(reader.fieldnames):
                raise ValueError("CSV needs 'class_id' and 'flower_name' headers")
            pairs = [(row["class_id"], row["flower_name"]) for row in reader]
    else:
        raise ValueError("Name mapping must be a .csv or .json file")

    out = {}
    for class_id, flower_name in pairs:
        class_id, flower_name = str(class_id).strip(), str(flower_name).strip()
        if not class_id or not flower_name:
            raise ValueError("Blank class ID or flower name found in mapping")
        if class_id in out:
            raise ValueError(f"Duplicate class ID in mapping: {class_id}")
        out[class_id] = flower_name
    return out


class FlowerClassifierExtended(FlowerClassifier):
    def __init__(self, model_path="flower_model_fp32.pt", device="cpu", names_path=None):
        super().__init__(model_path, device)
        self.flower_names = load_flower_names(names_path) if names_path is not None else {}
        unknown_ids = set(self.flower_names) - set(self.idx_to_class.values())
        if unknown_ids:
            raise ValueError(f"Name mapping contains unknown class IDs: {sorted(unknown_ids)}")

    @torch.inference_mode()
    def predict_details(self, path) -> dict:
        """One image -> class ID, optional name, max-softmax confidence (not accuracy)."""
        x = self._load(path).unsqueeze(0).to(self.device)
        x = (x - self.mean) / self.std
        if self.fp16:
            x = x.half()
        logits = self.model(x)
        probabilities = torch.softmax(logits.float(), dim=1)
        confidence, idx = probabilities.max(dim=1)
        class_id = self.idx_to_class[int(idx.item())]
        return {
            "class_id": class_id,
            "flower_name": self.flower_names.get(class_id),
            "confidence": float(confidence.item()),
        }

    def predict_name(self, path) -> str | None:
        return self.predict_details(path)["flower_name"]


def main():
    ap = argparse.ArgumentParser(description="Print class ID, optional flower name and confidence")
    ap.add_argument("model", help="TorchScript model exported by scripts/export.py")
    ap.add_argument("image", help="Path to a flower photo")
    ap.add_argument("--names", default=None, help="Path to results/class_to_name_mapping.csv or cat_to_name.json")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    classifier = FlowerClassifierExtended(args.model, device="cpu", names_path=args.names)
    print(json.dumps(classifier.predict_details(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
