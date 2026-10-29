"""KATMAN 7 - Giris noktasi. Amac: bir checkpoint'i degerlendirmek ve sinif bazli metrikleri kaydetmek.
Klasor yapisi: <root>/<sinif_adi>/*.jpg
Ornek: python scripts/evaluate.py --ckpt runs/mnv3_small/best.pt --data data/flowers/valid --out_csv results/per_class.csv
"""
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from flowers.export.packager import load_checkpoint_model
from flowers.data.transforms import build_eval_transform
from flowers.training.engine import predict_loader
from flowers.evaluation.metrics import top1_accuracy, per_class_accuracy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Model checkpoint yolu (.pt)")
    ap.add_argument("--data", required=True, help="Test veya valid klasor yolu")
    ap.add_argument("--bs", type=int, default=64, help="Batch size")
    ap.add_argument("--out_csv", type=str, default="results/per_class.csv", help="Sinif bazli dogruluk CSV cikis yolu")
    a = ap.parse_args()

    model, ck = load_checkpoint_model(a.ckpt)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_eval_transform(ck["img_size"], ck["crop_pct"], ck["data_cfg"]["mean"], ck["data_cfg"]["std"])
    ds = ImageFolder(a.data, transform=transform)

    remap = torch.tensor([ck["class_to_idx"][c] for c in ds.classes])

    logits, targets = predict_loader(model.to(dev), DataLoader(ds, batch_size=a.bs, num_workers=2), dev, channels_last=False)
    remapped_targets = remap[targets]

    acc = top1_accuracy(logits, remapped_targets)
    print(f"\n========================================")
    print(f"Top-1 Accuracy: {acc:.4f} (Toplam Gorsel: {len(ds)})")
    print(f"========================================\n")

    num_classes = len(ck["class_to_idx"])
    per_class_acc = per_class_accuracy(logits, remapped_targets, num_classes)

    idx_to_class = {v: k for k, v in ck["class_to_idx"].items()}
    records = []
    for idx in range(num_classes):
        records.append({
            "class_idx": idx,
            "class_name": idx_to_class.get(idx, f"class_{idx}"),
            "accuracy": round(per_class_acc[idx].item(), 4)
        })

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(a.out_csv, index=False)
    print(f"Sinif bazli dogruluk raporu kaydedildi -> {a.out_csv}")
