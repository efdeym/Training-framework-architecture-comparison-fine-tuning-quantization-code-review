"""C7 - export/benchmark zincirini denemek icin rastgele agirlikli gecici checkpoint uretir.

Notlar:
- Agirliklar rastgeledir (pretrained=False); bu checkpoint egitilmis model DEGILDIR.
- Yalnizca cikarim zincirinin (load -> export -> benchmark) calistigini gostermek icindir.
- Alanlar src/flowers/export/packager.py'nin bekledigi anahtarlardir.
Kullanim: python scratch/c7_make_dummy_ckpt.py <cikis_yolu>
"""
import argparse

import torch

from flowers.models.factory import create_model, get_data_config

ARCH = "mobilenetv3_small_100"
NUM_CLASSES = 102
IMG_SIZE = 224

parser = argparse.ArgumentParser(description="C7 - dummy checkpoint uret")
parser.add_argument("out", help="Yazilacak checkpoint yolu")
args = parser.parse_args()

model = create_model(ARCH, NUM_CLASSES, pretrained=False)
cfg = get_data_config(model)

torch.save(
    {
        "arch": ARCH,
        "num_classes": NUM_CLASSES,
        "model": model.state_dict(),
        "img_size": IMG_SIZE,
        "class_to_idx": {str(i): i for i in range(NUM_CLASSES)},
        "data_cfg": {"mean": cfg["mean"], "std": cfg["std"]},
        "crop_pct": 0.875,
    },
    args.out,
)
print("dummy checkpoint yazildi:", args.out)

