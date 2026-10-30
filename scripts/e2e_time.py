import sys, os, glob
root = os.getcwd()
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "src"))
sys.path.insert(0, os.path.join(root, "deliver"))   # predict.py burada
import torch
from predict import FlowerClassifier
from flowers.evaluation.benchmark import measure_end_to_end_ms

torch.set_num_threads(1)
model = sys.argv[1]
clf = FlowerClassifier(model, device="cpu")
paths = sorted(glob.glob(r"data\flowers\test\*\*.jpg"))[:200]
ms = measure_end_to_end_ms(clf.predict, paths)
b = os.path.getsize(model)
print(f"uctan uca: {ms:.2f} ms/goruntu ({len(paths)} gorsel, 1 thread, batch=1)")
print(f"MB: {b/1024**2:.2f} MiB | {b/1e6:.2f} MB (10^6)")