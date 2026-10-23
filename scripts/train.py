"""KATMAN 7 - Giriş noktası. Amaç: config ile eğitim başlatmak.
Örnek: python scripts/train.py --config configs/mobilenetv3.yaml train.epochs=20 exp_name=mnv3_a"""
import argparse
from flowers.config import load_config
from flowers.utils.seed import seed_everything
from flowers.training.trainer import Trainer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    seed_everything(cfg.seed)
    best = Trainer(cfg).fit()
    print(f"Bitti. best val_top1={best:.4f}")
