"""predict.py ön işlemesi ile eğitim eval transform'u yakın sonuç vermeli (sızıntı/uyumsuzluk kontrolü)."""
import sys, pathlib
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "deliver"))
from predict import FlowerClassifier
from flowers.data.transforms import build_eval_transform, IMAGENET_MEAN, IMAGENET_STD


def test_preprocess_matches(tmp_path):
    img = Image.fromarray((np.random.rand(180, 260, 3) * 255).astype("uint8")); p = tmp_path / "a.jpg"; img.save(p)
    clf = FlowerClassifier.__new__(FlowerClassifier)
    clf.size, clf.crop_pct = 224, 0.875
    a = clf._load(str(p))
    a = (a - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    b = build_eval_transform(224, 0.875)(Image.open(p).convert("RGB"))
    assert (a - b).abs().mean() < 0.05
