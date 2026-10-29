import hashlib, pathlib, sys
root = pathlib.Path("data/flowers")
def md5s(split):
    d = {}
    for p in (root / split).rglob("*.jpg"):
        d.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(str(p))
    return d
tr, va, te = md5s("train"), md5s("valid"), md5s("test")
for a, b, name in [(tr, te, "train-test"), (tr, va, "train-valid"), (va, te, "valid-test")]:
    common = set(a) & set(b)
    print(name, len(common))
    for h in list(common)[:10]:
        print("  ", a[h], "<->", b[h])