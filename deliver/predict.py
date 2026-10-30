"""KATMAN 6 - Teslim. Bağımsız tahmin dosyası: sadece torch, numpy, Pillow gerekir (timm GEREKMEZ).

Kullanım:
    from predict import FlowerClassifier
    clf = FlowerClassifier("flower_model.pt")           # device="cuda" da olabilir
    label = clf.predict("foto.jpg")                      # orijinal sınıf klasör adı (örn. "73")
    labels = clf.predict_batch(["a.jpg", "b.jpg"])
    top5 = clf.predict_topk("foto.jpg", k=5)             # [{"class": "73", "name": "...", "probability": 93.1}, ...]

Girdi doğrulama: dosya yoksa FileNotFoundError, dosya geçerli bir görsel değilse / bozuksa InvalidImageError
(ValueError alt sınıfı) fırlatılır.

Komut satırı (tek görsel ya da klasör):
    python predict.py --image foto.jpg
    python predict.py --image_dir gorseller/ --json
    python predict.py --model flower_model.pt --image foto.jpg --topk 5 --device cpu
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class InvalidImageError(ValueError):
    """Dosya var ama geçerli bir görsel değil (bozuk ya da görsel olmayan içerik)."""


def load_image(path) -> Image.Image:
    """Girdi doğrulamalı görsel okuma: (a) dosya var mı, (b) geçerli/bozuk görsel mi."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Görsel bulunamadı: {path}")
    if not os.path.isfile(path):
        raise InvalidImageError(f"Yol bir dosya değil: {path}")
    try:
        img = Image.open(path)
        img.load()  # kesik/bozuk veriyi burada yakalamak için
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"Geçerli bir görsel değil veya bozuk: {path} ({type(e).__name__})") from e


class FlowerClassifier:
    def __init__(self, model_path: str = "flower_model.pt", device: str = "cpu", names_path: str = None):
        extra = {"meta.json": ""}
        self.device = torch.device(device)
        # Dosya nesnesi ile yükle: Windows'ta ASCII olmayan (örn. "ü", "ı") yollarda yol ile yüklemek başarısız olabiliyor
        with open(model_path, "rb") as f:
            self.model = torch.jit.load(f, map_location=self.device, _extra_files=extra).eval()
        self.meta = json.loads(extra["meta.json"])
        self.idx_to_class = {int(k): v for k, v in self.meta["idx_to_class"].items()}
        self.fp16 = self.meta["dtype"] == "fp16" and self.device.type == "cuda"
        if self.meta["dtype"] == "fp16" and not self.fp16:
            self.model = self.model.float()      # CPU'da fp16 hızlı değil, fp32'ye çevir
        self.mean = torch.tensor(self.meta["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.meta["std"], device=self.device).view(1, 3, 1, 1)
        self.size, self.crop_pct = self.meta["img_size"], self.meta["crop_pct"]
        self.norm_in_model = bool(self.meta.get("norm_in_model", False))
        # Sınıf adları (opsiyonel): klasör adı -> çiçek adı. Verilmezse predict.py'nin yanındaki dosya aranır.
        names_path = names_path or os.path.join(HERE, "cat_to_name.json")
        self.names = {}
        if os.path.isfile(names_path):
            with open(names_path, encoding="utf-8") as f:
                self.names = json.load(f)

    def _load(self, path) -> torch.Tensor:
        img = load_image(path)
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
        return x.contiguous()

    def _logits(self, paths) -> torch.Tensor:
        x = torch.stack([self._load(p) for p in paths]).to(self.device)
        if not self.norm_in_model:
            x = (x - self.mean) / self.std
        if self.fp16:
            x = x.half()
        return self.model(x)

    @torch.inference_mode()
    def predict_batch(self, paths):
        idx = self._logits(paths).argmax(1).tolist()
        return [self.idx_to_class[i] for i in idx]

    def predict(self, path) -> str:
        return self.predict_batch([path])[0]

    @torch.inference_mode()
    def predict_topk(self, path, k: int = 5):
        """En olası k sınıf: [{"class": klasör adı, "name": çiçek adı (yoksa None), "probability": yüzde}, ...]"""
        probs = torch.softmax(self._logits([path]).float(), dim=1)[0]
        top_p, top_i = probs.topk(min(k, probs.numel()))
        out = []
        for p, i in zip(top_p.tolist(), top_i.tolist()):
            cls = self.idx_to_class[i]
            out.append({"class": cls, "name": self.names.get(cls), "probability": round(p * 100, 2)})
        return out


def _list_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def _print_text(path, result):
    print(path)
    if "error" in result:
        print(f"  HATA ({result['error_type']}): {result['error']}")
        return
    for rank, r in enumerate(result["top"], start=1):
        print(f"  {rank}. %{r['probability']:6.2f}  sınıf={r['class']:>3}  {r['name'] or '-'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Çiçek sınıflandırma: tek görsel ya da klasör için Top-K tahmin")
    ap.add_argument("--model", default=os.path.join(HERE, "flower_model.pt"), help="TorchScript model yolu")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Tek görsel yolu")
    g.add_argument("--image_dir", help="Görsellerin bulunduğu klasör")
    ap.add_argument("--topk", type=int, default=5, help="Gösterilecek sınıf sayısı (varsayılan 5)")
    ap.add_argument("--device", default="cpu", help="cpu veya cuda")
    ap.add_argument("--names", default=None, help="cat_to_name.json yolu (varsayılan: predict.py'nin yanı)")
    ap.add_argument("--json", action="store_true", help="Çıktıyı JSON olarak yazdır")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.model):
        print(f"Model bulunamadı: {a.model}", file=sys.stderr)
        return 2
    if a.image_dir is not None:
        if not os.path.isdir(a.image_dir):
            print(f"Klasör bulunamadı: {a.image_dir}", file=sys.stderr)
            return 2
        paths = _list_images(a.image_dir)
        if not paths:
            print(f"Klasörde desteklenen görsel yok ({', '.join(sorted(IMAGE_EXTS))}): {a.image_dir}", file=sys.stderr)
            return 2
    else:
        paths = [a.image]

    clf = FlowerClassifier(a.model, device=a.device, names_path=a.names)
    results, failed = {}, 0
    for p in paths:
        try:
            results[p] = {"top": clf.predict_topk(p, a.topk)}
        except (FileNotFoundError, InvalidImageError) as e:
            failed += 1
            results[p] = {"error_type": type(e).__name__, "error": str(e)}

    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for p, r in results.items():
            _print_text(p, r)
        if len(paths) > 1:
            print(f"\nToplam {len(paths)} görsel, {len(paths) - failed} başarılı, {failed} hatalı.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
