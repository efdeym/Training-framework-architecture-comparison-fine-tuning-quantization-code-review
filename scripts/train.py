# """KATMAN 7 - Giriş noktası. Amaç: config ile eğitim başlatmak.
# Örnek: python scripts/train.py --config configs/mobilenetv3.yaml train.epochs=20 exp_name=mnv3_a"""
# import argparse
# from flowers.config import load_config
# from flowers.utils.seed import seed_everything
# from flowers.training.trainer import Trainer

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("overrides", nargs="*")
#     a = ap.parse_args()
#     cfg = load_config(a.config, a.overrides)
#     seed_everything(cfg.seed)
#     best = Trainer(cfg).fit()
#     print(f"Bitti. best val_top1={best:.4f}")

"""KATMAN 7 - Giriş noktası. Amaç: config ile eğitim başlatmak.
Örnek: python scripts/train.py --config configs/mobilenetv3.yaml train.epochs=20 exp_name=mnv3_a"""
import argparse
import sys
from pathlib import Path
from flowers.config import load_config
from flowers.utils.seed import seed_everything
from flowers.training.trainer import Trainer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Var olan exp_name klasorunun uzerine yazmaya izin ver")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)

    run_dir = Path("runs") / cfg.exp_name
    if not a.force and any((run_dir / f).exists() for f in ("best.pt", "last.pt", "train.log")):
        sys.exit(f"HATA: {run_dir} zaten dolu. Yeni bir exp_name ver "
                 f"(ornek: {cfg.exp_name}_v2) ya da bilerek ezmek icin --force ekle.")

    seed_everything(cfg.seed)
    best = Trainer(cfg).fit()
    print(f"Bitti. best val_top1={best:.4f}")