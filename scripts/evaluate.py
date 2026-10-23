"""KATMAN 7 - Giriş noktası. Amaç: bir checkpoint'i herhangi bir klasörde (örn. shadow_test/) değerlendirmek.
Klasör yapısı: <root>/<sınıf_adı>/*.jpg  (sınıf adları orijinal klasör adlarıyla aynı olmalı)
Örnek: python scripts/evaluate.py --ckpt runs/exp/best.pt --data data/shadow_test"""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from flowers.export.packager import load_checkpoint_model
from flowers.data.transforms import build_eval_transform
from flowers.training.engine import predict_loader
from flowers.evaluation.metrics import top1_accuracy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    model, ck = load_checkpoint_model(a.ckpt)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ImageFolder(a.data, build_eval_transform(ck["img_size"], ck["crop_pct"], ck["data_cfg"]["mean"], ck["data_cfg"]["std"]))
    # klasör adı -> eğitimdeki indekse çevir (gölge sette bazı sınıflar eksik olabilir)
    remap = torch.tensor([ck["class_to_idx"][c] for c in ds.classes])
    logits, targets = predict_loader(model.to(dev), DataLoader(ds, batch_size=a.bs, num_workers=4), dev, channels_last=False)
    print(f"top1 = {top1_accuracy(logits, remap[targets]):.4f}  (n={len(ds)})")
