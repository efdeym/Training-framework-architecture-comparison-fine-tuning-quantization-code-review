import sys, glob, os, torch

sys.path.insert(0, "deliver")
from predict import FlowerClassifier

torch.set_num_threads(4)
clf = FlowerClassifier(sys.argv[1])
paths = sorted(glob.glob("data/flowers/valid/*/*.jpg"))
labels = [os.path.basename(os.path.dirname(p)) for p in paths]

modes = {
    "ham [0,1]": lambda x: x,
    "ImageNet normalize edilmis": lambda x: (x - clf.mean) / clf.std,
}

for name, fn in modes.items():
    ok = 0
    for i in range(0, len(paths), 32):
        x = torch.stack([clf._load(p) for p in paths[i:i + 32]])
        with torch.inference_mode():
            pred = clf.model(fn(x)).argmax(1).tolist()
        ok += sum(clf.idx_to_class[a] == b for a, b in zip(pred, labels[i:i + 32]))
    print(f"{name:28s} %{100 * ok / len(paths):.2f} ({ok}/{len(paths)})")