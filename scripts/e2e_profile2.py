import sys, glob, random, time, torch
sys.path.insert(0, "deliver")
from predict import FlowerClassifier

torch.set_num_threads(1)
c = FlowerClassifier("deliver/flower_model.pt")
random.seed(0)
paths = random.sample(glob.glob("data/flowers/test/*/*.jpg"), 100)
xs = [c._load(p).unsqueeze(0) for p in paths]

def bench(items, name, warm=10):
    with torch.inference_mode():
        for x in items[:warm]:
            c.model(x)
        t0 = time.perf_counter()
        for x in items:
            c.model(x)
    print(f"{name:34s} {(time.perf_counter() - t0) * 1000 / len(items):.1f} ms")

for tur in (1, 2):
    bench([torch.randn(1, 3, 224, 224) for _ in range(100)], "randn (benchmark.py gibi)")
    bench([torch.rand(1, 3, 224, 224) for _ in range(100)], "rand [0,1]")
    bench([x.contiguous() for x in xs], "gercek gorsel, contiguous")
    bench(xs, "gercek gorsel, _load ciktisi")
print("strides _load:", xs[0].stride(), "| contiguous:", xs[0].is_contiguous())