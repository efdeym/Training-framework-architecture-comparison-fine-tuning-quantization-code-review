import glob, os, sys
import numpy as np
import torch
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"submission\model.pt"
m = torch.jit.load(MODEL, map_location="cpu").eval()   # extra_files yok, hoca gibi


def prep(path, resample):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nh, nw = 256, int(256 * w / h)
    im = im.resize((nw, nh), resample)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


for name, rs in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
    for split in ("valid", "test"):
        paths = sorted(glob.glob(rf"data\flowers\{split}\*\*.jpg"))
        ok = 0
        for i in range(0, len(paths), 32):
            b = paths[i:i + 32]
            x = torch.stack([prep(p, rs) for p in b])
            with torch.no_grad():
                y = m(x)
            assert y.dtype == torch.float32 and tuple(y.shape) == (len(b), 102), y.shape
            assert float(y.min()) >= 0 and float(y.max()) <= 1
            assert torch.allclose(y.sum(1), torch.ones(len(b)), atol=1e-4)
            true = torch.tensor([int(os.path.basename(os.path.dirname(p))) for p in b])
            ok += int(((y.argmax(1) + 1) == true).sum())
        print(f"{name} {split}: {ok}/{len(paths)} = {ok/len(paths):.4f}")
print("OK: shape, range, sum, and order checks passed")