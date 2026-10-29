"""Checkpoint ile TorchScript aynı sonucu veriyor mu? Kullanım: python scripts/verify_export.py CKPT PT"""
import sys, glob, os, random, torch
sys.path.insert(0, "deliver")
from predict import FlowerClassifier
from flowers.export.packager import load_checkpoint_model

ckpt, pt = sys.argv[1], sys.argv[2]
model, ck = load_checkpoint_model(ckpt)
clf = FlowerClassifier(pt)
mean = torch.tensor(ck["data_cfg"]["mean"]).view(1, 3, 1, 1)
std = torch.tensor(ck["data_cfg"]["std"]).view(1, 3, 1, 1)
random.seed(0)
paths = random.sample(glob.glob("data/flowers/valid/*/*.jpg"), 10)
same, correct, worst = 0, 0, 0.0
for p in paths:
    x = clf._load(p).unsqueeze(0)
    with torch.inference_mode():
        a = model((x - mean) / std)
        b = clf._logits([p])
    worst = max(worst, (a - b).abs().max().item())
    same += int(a.argmax(1).item() == b.argmax(1).item())
    correct += int(clf.idx_to_class[b.argmax(1).item()] == os.path.basename(os.path.dirname(p)))
print(f"ayni sinif: {same}/10 | dogru: {correct}/10 | max logit farki: {worst:.2e}")
with torch.inference_mode():
    print("batch=4 cikis:", tuple(clf._logits(paths[:4]).shape))