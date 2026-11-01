import glob, os, sys, time
from collections import defaultdict
import numpy as np
import torch
from PIL import Image

MODEL = sys.argv[1]
DATA = sys.argv[2]                                   # alt klasorler = sinif numaralari
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)

model = torch.jit.load(MODEL, map_location="cpu").eval()


def prep(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nh, nw = 256, int(256 * w / h)
    im = im.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


paths = []
for ext in ("jpg", "jpeg", "png"):
    paths += glob.glob(os.path.join(DATA, "*", f"*.{ext}"))
paths = sorted(paths)

with torch.no_grad():
    for _ in range(5):
        model(torch.rand(1, 3, 224, 224))            # isinma

ok, t_prep, t_fwd = 0, 0.0, 0.0
wrong = []
per = defaultdict(lambda: [0, 0])
for p in paths:
    t0 = time.perf_counter()
    x = prep(p)
    t1 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    t2 = time.perf_counter()
    pred = int(y.argmax(1)) + 1
    true = int(os.path.basename(os.path.dirname(p)))
    per[true][1] += 1
    per[true][0] += pred == true
    ok += pred == true
    if pred != true:
        wrong.append((p, true, pred))
    t_prep += t1 - t0
    t_fwd += t2 - t1

n = len(paths)
acc = ok / n
fwd_ms, e2e_ms = t_fwd / n * 1000, (t_fwd + t_prep) / n * 1000
mb = os.path.getsize(MODEL) / 1e6


def score(t_ms):
    return 0.60 * acc * 100 + 0.25 * min(100, 50 / t_ms * 100) + 0.15 * min(100, 35 / mb * 100)


print(f"gorsel: {n} | dogru: {ok} | top-1: {acc:.4f}")
print("sinif bazli (dogru/toplam):")
for c in sorted(per):
    print(f"  sinif {c}: {per[c][0]}/{per[c][1]}")
print(f"model suresi: {fwd_ms:.2f} ms | on isleme dahil: {e2e_ms:.2f} ms | boyut: {mb:.2f} MB")
print(f"tahmini skor (model suresi): {score(fwd_ms):.2f} | (on isleme dahil): {score(e2e_ms):.2f}")
print("ilk 15 hata:")
for p, t, pr in wrong[:15]:
    print(f"  {p} gercek={t} tahmin={pr}")