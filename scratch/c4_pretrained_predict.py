"""C4 - timm ile onceden egitilmis mobilenetv3_small_100 yukle, tek gorselde tahmin yap.

Notlar:
- pretrained=True internetten (Hugging Face Hub) agirlik indirir.
- Model ImageNet-1k (1000 sinif) uzerinde egitilmistir, cicek veri seti icin egitilmedi.
  Bu yuzden cikan sinif indeksi bir cicek sinifi degildir; script yalnizca boru hattinin calistigini gosterir.
- On isleme timm.data.resolve_data_config + create_transform ile alinir.
"""
import argparse
import sys
from pathlib import Path

import timm
import torch
from PIL import Image

parser = argparse.ArgumentParser(description="C4 - tek gorselde pretrained tahmin")
parser.add_argument("image", help="Tahmin yapilacak gorselin yolu")
IMAGE_PATH = Path(parser.parse_args().image)
MODEL_NAME = "mobilenetv3_small_100"

if not IMAGE_PATH.is_file():
    sys.exit(f"Gorsel bulunamadi: {IMAGE_PATH}")

print("torch:", torch.__version__, "| timm:", timm.__version__)
print("gorsel:", IMAGE_PATH)

try:
    model = timm.create_model(MODEL_NAME, pretrained=True)
except Exception as e:  # indirme hatasini oldugu gibi goster
    print("Model yuklenemedi:", type(e).__name__)
    print(e)
    sys.exit(1)

model.eval()

data_cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_cfg)
print("data_cfg:", data_cfg)
print("transform:", transform)

img = Image.open(IMAGE_PATH).convert("RGB")
print("orijinal gorsel boyutu (w, h):", img.size)

x = transform(img).unsqueeze(0)  # [1, 3, H, W]
print("model girdisi shape:", tuple(x.shape), "| dtype:", x.dtype, "| model.training:", model.training)

with torch.inference_mode():
    logits = model(x)

print("cikti shape:", tuple(logits.shape), "| dtype:", logits.dtype)
pred = logits.argmax(1).item()
print("tahmin edilen ImageNet sinif indeksi (0-999):", pred)
print("UYARI: bu indeks ImageNet'e aittir, cicek sinifi degildir (C5'te top-5 yazdirilacak).")
