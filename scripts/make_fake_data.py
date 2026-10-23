"""Duman testi için sahte veri üretir (gerçek veri yokken pipeline'ı denemek için)."""
import sys, random
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fake"); n_cls = 4
for split, n in [("train", 6), ("valid", 2), ("test", 2)]:
    for c in ["1", "10", "2", "100"][:n_cls]:
        d = root / split / c; d.mkdir(parents=True, exist_ok=True)
        base = random.Random(c).choices(range(256), k=3)
        for i in range(n):
            Image.new("RGB", (160, 140), tuple((b + random.randint(-20, 20)) % 256 for b in base)).save(d / f"{i}.jpg")
print("ok", root)
