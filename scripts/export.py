"""KATMAN 7 - Giriş noktası. Amaç: checkpoint -> tek dosya TorchScript (+ gömülü etiket eşlemesi).
Örnek: python scripts/export.py --ckpt runs/exp/last.pt --out flower_model.pt --dtype fp32"""
import argparse
from flowers.export.packager import load_checkpoint_model, export_torchscript
from flowers.evaluation.benchmark import model_size_mb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="flower_model.pt")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    a = ap.parse_args()
    model, ck = load_checkpoint_model(a.ckpt)
    meta = export_torchscript(model, ck, a.out, a.dtype)
    print(f"Kaydedildi: {a.out}  ({model_size_mb(a.out):.2f} MB)  arch={meta['arch']} dtype={a.dtype}")
