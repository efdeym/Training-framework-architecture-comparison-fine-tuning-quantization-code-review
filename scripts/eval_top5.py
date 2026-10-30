import sys, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

model_path, data_dir = sys.argv[1], Path(sys.argv[2])
extra = {"meta.json": ""}
model = torch.jit.load(model_path, map_location="cpu", _extra_files=extra).eval()
meta = json.loads(extra["meta.json"])
size, crop_pct = meta["img_size"], meta["crop_pct"]
idx2cls = {int(k): v for k, v in meta["idx_to_class"].items()}
norm_in = meta.get("norm_in_model", False)
MEAN = torch.tensor(meta["mean"]).view(3, 1, 1)
STD = torch.tensor(meta["std"]).view(3, 1, 1)

def prep(p):
    im = Image.open(p).convert("RGB")
    s = int(size / crop_pct)
    w, h = im.size
    nw, nh = (s, round(h * s / w)) if w < h else (round(w * s / h), s)
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - size) // 2, (nh - size) // 2
    im = im.crop((l, t, l + size, t + size))
    a = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).contiguous()
    return t if norm_in else ((t - MEAN) / STD).contiguous()

items = [(p, d.name) for d in sorted(data_dir.iterdir()) if d.is_dir()
         for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
c1 = c5 = 0
per = {}
for i in range(0, len(items), 32):
    chunk = items[i:i + 32]
    x = torch.stack([prep(p) for p, _ in chunk])
    with torch.inference_mode():
        top = model(x).topk(5, dim=1).indices.tolist()
    for (p, true), t in zip(chunk, top):
        names = [idx2cls[j] for j in t]
        ok1, ok5 = names[0] == true, true in names
        c1 += ok1; c5 += ok5
        a = per.setdefault(true, [0, 0]); a[0] += ok1; a[1] += 1
n = len(items)
zero = sum(1 for a, b in per.values() if a == 0)
print(f"{Path(model_path).name} | {data_dir} | n={n} sinif={len(per)} "
      f"top1={c1/n*100:.2f} top5={c5/n*100:.2f} sifir_dogruluklu_sinif={zero}")