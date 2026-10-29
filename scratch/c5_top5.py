"""C5 - softmax + topk ile top-5 sinif indeksi ve olasiligi.

Notlar:
- mobilenetv3_small_100.lamb_in1k ImageNet-1k (1000 sinif) uzerinde egitilmistir; 102 cicek sinifi icin DEGIL.
  Sonuclar yalnizca boru hattinin calistigini gosterir, cicek dogrulugu hakkinda bilgi vermez.
- Sinif adlari internet gerektirmeden timm paketinin icindeki dosyalardan (timm/data/_info) okunur.
  Ad okunamazsa yalnizca indeks yazdirilir ve ad dogrulanmadi diye belirtilir.
"""
import argparse
import sys
from pathlib import Path

import timm
import torch
from PIL import Image

parser = argparse.ArgumentParser(description="C5 - top-5 tahmin")
parser.add_argument("image", help="Tahmin yapilacak gorselin yolu")
IMAGE_PATH = Path(parser.parse_args().image)
MODEL_NAME = "mobilenetv3_small_100"
K = 5

if not IMAGE_PATH.is_file():
    sys.exit(f"Gorsel bulunamadi: {IMAGE_PATH}")

print("torch:", torch.__version__, "| timm:", timm.__version__)
print("gorsel:", IMAGE_PATH)

try:
    model = timm.create_model(MODEL_NAME, pretrained=True)
except Exception as e:
    print("Model yuklenemedi:", type(e).__name__)
    print(e)
    sys.exit(1)
model.eval()

data_cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_cfg)

img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0)

with torch.inference_mode():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)

print("logits shape:", tuple(logits.shape), "| softmax toplami:", round(probs.sum().item(), 6))

top_p, top_i = probs.topk(K, dim=1)

# Sinif adlari: timm icindeki yerel dosyalar (internet yok)
info = None
try:
    from timm.data.imagenet_info import ImageNetInfo
    info = ImageNetInfo("imagenet-1k")
except Exception as e:
    print("Sinif adi kaynagi yuklenemedi:", type(e).__name__, e)

print(f"\nTop-{K} (ImageNet-1k):")
for rank, (p, i) in enumerate(zip(top_p[0].tolist(), top_i[0].tolist()), start=1):
    name = "ad dogrulanmadi"
    synset = "-"
    if info is not None:
        try:
            synset = info.index_to_label_name(i)
            name = info.index_to_description(i)
        except Exception as e:
            name = f"ad okunamadi ({type(e).__name__}: {e})"
    print(f"{rank}. indeks={i:4d}  olasilik={p:.4f}  synset={synset}  ad={name}")

print(f"\nTop-{K} olasilik toplami: {top_p.sum().item():.4f}")
print("UYARI: bunlar ImageNet siniflaridir, cicek sinifi degildir; model cicek veri seti icin egitilmedi.")
