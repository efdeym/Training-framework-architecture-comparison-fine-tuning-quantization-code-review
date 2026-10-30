import sys, os, glob
root = os.getcwd()
sys.path.insert(0, os.path.join(root, "deliver"))
from predict import FlowerClassifier

clf = FlowerClassifier(r"deliver\flower_model.pt", device="cpu")
for split in ("valid", "test"):
    paths = glob.glob(rf"data\flowers\{split}\*\*.jpg")
    ok = sum(clf.predict(p) == os.path.basename(os.path.dirname(p)) for p in paths)
    print(f"{split}: {ok}/{len(paths)} = {ok/len(paths):.4f}")