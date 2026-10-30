import sys, glob, random, time
import torch
sys.path.insert(0, "deliver")
from predict import FlowerClassifier, load_image

torch.set_num_threads(1)
c = FlowerClassifier("deliver/flower_model.pt")
random.seed(0)
paths = random.sample(glob.glob("data/flowers/test/*/*.jpg"), 200)
for p in paths:
    open(p, "rb").read()          # disk onbellegini isit

def t(fn, items):
    t0 = time.perf_counter()
    out = [fn(i) for i in items]
    return (time.perf_counter() - t0) * 1000 / len(items), out

for tur in (1, 2):
    a, _ = t(load_image, paths)
    b, xs = t(c._load, paths)
    with torch.inference_mode():
        m, _ = t(lambda x: c.model(x.unsqueeze(0)), xs)
    full, _ = t(c.predict, paths)
    print(f"tur {tur}: decode+RGB {a:.1f} | on isleme toplam {b:.1f} | model {m:.1f} | predict() {full:.1f}  ms/gorsel")