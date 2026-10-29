"""C9/C10 - predict(image_path, model, class_to_idx, device) taslagi + girdi dogrulama.

Notlar:
- deliver/predict.py'ye DOKUNULMADI; bu bir taslaktir.
- Dogrulama: (a) dosya var mi (os.path.exists), (b) gecerli/bozuk gorsel mi (PIL).
- On isleme sabitleri (224, crop_pct 0.875, ImageNet mean/std) deliver/predict.py'deki mantigin
  kopyasidir; gercek entegrasyonda bunlar modelin meta.json'undan okunmali.
- Duman testindeki model rastgele agirlikli kucuk bir modeldir; sonuc anlamsizdir.
Kullanim: python scratch/c9_c10_predict_draft.py
"""
import os
import tempfile

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

SIZE, CROP_PCT = 224, 0.875
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class InvalidImageError(ValueError):
    """Dosya var ama gecerli bir gorsel degil (bozuk ya da gorsel olmayan icerik)."""


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gorsel bulunamadi: {image_path}")
    if not os.path.isfile(image_path):
        raise InvalidImageError(f"Yol bir dosya degil: {image_path}")
    try:
        img = Image.open(image_path)
        img.load()  # kesik/bozuk veriyi burada yakalamak icin
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"Gecerli bir gorsel degil veya bozuk: {image_path} ({type(e).__name__})") from e


def preprocess(img):
    resize_to = int(SIZE / CROP_PCT)
    w, h = img.size
    if w <= h:
        nw, nh = resize_to, int(resize_to * h / w)
    else:
        nh, nw = resize_to, int(resize_to * w / h)
    img = img.resize((nw, nh), Image.BICUBIC)
    w, h = img.size
    l, t = int(round((w - SIZE) / 2.0)), int(round((h - SIZE) / 2.0))
    img = img.crop((l, t, l + SIZE, t + SIZE))
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def predict(image_path, model, class_to_idx, device="cpu"):
    """Tek gorsel icin sinif adini dondurur. Gecersiz girdide FileNotFoundError / InvalidImageError."""
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    x = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    idx = model.to(device).eval()(x).argmax(1).item()
    return idx_to_class[idx]


if __name__ == "__main__":
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4))
    class_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    good = r"C:\dev\data\flowers\valid\1\image_06739.jpg"

    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bozuk.jpg")
        with open(bad, "wb") as f:
            f.write(b"bu bir gorsel degil")
        truncated = os.path.join(tmp, "kesik.jpg")
        with open(good, "rb") as f:
            data = f.read()
        with open(truncated, "wb") as f:
            f.write(data[: len(data) // 3])

        cases = [
            ("1) olmayan dosya", os.path.join(tmp, "yok.jpg")),
            ("2) gorsel olmayan icerik (.jpg uzantili)", bad),
            ("3) kesilmis gorsel dosyasi", truncated),
            ("4) klasor yolu", tmp),
            ("5) gecerli gorsel", good),
        ]
        for name, path in cases:
            try:
                print(name, "->", "SONUC:", predict(path, model, class_to_idx))
            except (FileNotFoundError, InvalidImageError) as e:
                print(name, "->", type(e).__name__ + ":", e)

