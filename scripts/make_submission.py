# import json, os
# import torch
# import torch.nn as nn

# SRC = r"deliver\flower_model.pt"
# OUT = r"submission\model.pt"

# extra = {"meta.json": ""}
# base = torch.jit.load(SRC, map_location="cpu", _extra_files=extra).eval()
# meta = json.loads(extra["meta.json"])
# print("meta:", {k: v for k, v in meta.items() if k != "idx_to_class"})
# print("kisa kenar:", int(meta["img_size"] / meta["crop_pct"]), "(must be 256 )")

# idx_to_class = {int(k): str(v) for k, v in meta["idx_to_class"].items()}
# class_to_idx = {v: k for k, v in idx_to_class.items()}
# assert len(idx_to_class) == 102
# assert sorted(int(c) for c in class_to_idx) == list(range(1, 103))
# # numerik sira: pozisyon n-1 = sinif n  ->  modelin ilgili sutununu al
# perm = torch.tensor([class_to_idx[str(n)] for n in range(1, 103)], dtype=torch.long)


# class Submission(nn.Module):
#     def __init__(self, base, perm):
#         super().__init__()
#         self.base = base
#         self.register_buffer("perm", perm)

#     def forward(self, x):
#         p = torch.softmax(self.base(x), dim=1)
#         return p.index_select(1, self.perm)


# m = Submission(base, perm).eval()
# try:
#     out = torch.jit.script(m)
# except Exception as e:
#     print("script olmadi, trace deneniyor:", e)
#     out = torch.jit.trace(m, torch.rand(1, 3, 224, 224))

# os.makedirs("submission", exist_ok=True)
# out.save(OUT)
# print("kaydedildi:", OUT, os.path.getsize(OUT) / 1e6, "MB")

import json, os, sys
import torch
import torch.nn as nn

SRC = sys.argv[1] if len(sys.argv) > 1 else r"deliver\flower_model.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"submission\model.pt"

extra = {"meta.json": ""}
base = torch.jit.load(SRC, map_location="cpu", _extra_files=extra).eval()
meta = json.loads(extra["meta.json"])
print("meta:", {k: v for k, v in meta.items() if k != "idx_to_class"})
print("kisa kenar:", int(meta["img_size"] / meta["crop_pct"]), "(hoca 256 istiyor)")

idx_to_class = {int(k): str(v) for k, v in meta["idx_to_class"].items()}
class_to_idx = {v: k for k, v in idx_to_class.items()}
assert len(idx_to_class) == 102
assert sorted(int(c) for c in class_to_idx) == list(range(1, 103))
perm = torch.tensor([class_to_idx[str(n)] for n in range(1, 103)], dtype=torch.long)


class Submission(nn.Module):
    def __init__(self, base, perm):
        super().__init__()
        self.base = base
        self.register_buffer("perm", perm)

    def forward(self, x):
        p = torch.softmax(self.base(x), dim=1)
        return p.index_select(1, self.perm)


m = Submission(base, perm).eval()
try:
    out = torch.jit.script(m)
except Exception as e:
    print("script olmadi, trace deneniyor:", e)
    out = torch.jit.trace(m, torch.rand(1, 3, 224, 224))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
out.save(OUT)
print("kaydedildi:", OUT, os.path.getsize(OUT) / 1e6, "MB")