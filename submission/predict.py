"""KATMAN 6 - Teslim. Bağımsız tahmin dosyası: sadece torch, numpy, Pillow gerekir (timm GEREKMEZ).

Kullanım:
    from predict import FlowerClassifier
    clf = FlowerClassifier("flower_model.pt")           # device="cuda" da olabilir
    label = clf.predict("foto.jpg")                      # orijinal sınıf klasör adı (örn. "73")
    labels = clf.predict_batch(["a.jpg", "b.jpg"])

Komut satırı:  python predict.py flower_model.pt foto.jpg
"""
import json, sys
import numpy as np
import torch
from PIL import Image


class FlowerClassifier:
    def __init__(self, model_path: str = "flower_model.pt", device: str = "cpu"):
        extra = {"meta.json": ""}
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device, _extra_files=extra).eval()
        self.meta = json.loads(extra["meta.json"])
        self.idx_to_class = {int(k): v for k, v in self.meta["idx_to_class"].items()}
        self.fp16 = self.meta["dtype"] == "fp16" and self.device.type == "cuda"
        if self.meta["dtype"] == "fp16" and not self.fp16:
            self.model = self.model.float()      # CPU'da fp16 hızlı değil, fp32'ye çevir
        self.mean = torch.tensor(self.meta["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.meta["std"], device=self.device).view(1, 3, 1, 1)
        self.size, self.crop_pct = self.meta["img_size"], self.meta["crop_pct"]

    def _load(self, path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        # Eğitimdeki eval transform ile aynı: kısa kenarı size/crop_pct yap -> merkez kırp
        resize_to = int(self.size / self.crop_pct)
        w, h = img.size
        # torchvision.Resize ile aynı boyut kuralı (uzun kenar int() ile kesilir)
        if w <= h:
            nw, nh = resize_to, int(resize_to * h / w)
        else:
            nh, nw = resize_to, int(resize_to * w / h)
        img = img.resize((nw, nh), Image.BICUBIC)
        w, h = img.size
        l, t = int(round((w - self.size) / 2.0)), int(round((h - self.size) / 2.0))
        img = img.crop((l, t, l + self.size, t + self.size))
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return x

    @torch.inference_mode()
    def predict_batch(self, paths):
        x = torch.stack([self._load(p) for p in paths]).to(self.device)
        x = (x - self.mean) / self.std
        if self.fp16:
            x = x.half()
        idx = self.model(x).argmax(1).tolist()
        return [self.idx_to_class[i] for i in idx]

    def predict(self, path) -> str:
        return self.predict_batch([path])[0]


if __name__ == "__main__":
    clf = FlowerClassifier(sys.argv[1])
    print(clf.predict(sys.argv[2]))
