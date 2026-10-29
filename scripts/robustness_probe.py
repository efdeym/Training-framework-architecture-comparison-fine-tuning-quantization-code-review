"""Ön işleme ve bozulma dayanıklılığı (valid). Kullanım: python scripts/robustness_probe.py MODEL.pt"""
import sys, os, io, glob, torch
sys.path.insert(0, "deliver")
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from predict import FlowerClassifier, load_image

torch.set_num_threads(4)
clf = FlowerClassifier(sys.argv[1])
S = clf.size

def center(img, short):
    w, h = img.size
    nw, nh = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
    img = img.resize((nw, nh), Image.BICUBIC)
    l, t = int(round((nw - S) / 2.0)), int(round((nh - S) / 2.0))
    return img.crop((l, t, l + S, t + S))

def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")

default = lambda im: center(im, int(S / clf.crop_pct))
variants = {
    "varsayilan (resize+crop)": default,
    "crop_pct=1.0 (resize 224+crop)": lambda im: center(im, S),
    "sikistirma (224x224 squash)": lambda im: im.resize((S, S), Image.BICUBIC),
    "bulaniklik (sigma=2)": lambda im: default(im.filter(ImageFilter.GaussianBlur(2))),
    "karanlik (x0.6)": lambda im: default(ImageEnhance.Brightness(im).enhance(0.6)),
    "parlak (x1.4)": lambda im: default(ImageEnhance.Brightness(im).enhance(1.4)),
    "jpeg kalite 20": lambda im: default(jpeg(im, 20)),
}
paths = sorted(glob.glob("data/flowers/valid/*/*.jpg"))
labels = [os.path.basename(os.path.dirname(p)) for p in paths]
imgs = [load_image(p) for p in paths]
for name, fn in variants.items():
    correct = 0
    for i in range(0, len(imgs), 64):
        x = torch.stack([torch.from_numpy(np.asarray(fn(im), dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
                         for im in imgs[i:i + 64]])
        with torch.inference_mode():
            pred = clf.model(x).argmax(1).tolist()
        correct += sum(clf.idx_to_class[p] == l for p, l in zip(pred, labels[i:i + 64]))
    print(f"{name:34s} %{100 * correct / len(paths):.2f}  ({correct}/{len(paths)})")