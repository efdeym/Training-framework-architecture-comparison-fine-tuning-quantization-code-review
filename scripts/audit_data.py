"""flower_data train/valid/test sayıları ve tam dosya kopyalarını kontrol et."""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(root):
    splits = ("train", "valid", "test")
    hashes = defaultdict(list)
    counts = {}
    for split in splits:
        d = root / split
        if not d.is_dir():
            print(f"{split}: klasör yok")
            continue
        cnt = Counter()
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in EXTS:
                continue
            cls = f.parent.name
            cnt[cls] += 1
            hashes[hashlib.sha256(f.read_bytes()).hexdigest()].append((split, cls, str(f)))
        counts[split] = cnt
        print(f"{split}: {sum(cnt.values())} görsel, {len(cnt)} sınıf, "
              f"min={min(cnt.values()) if cnt else 0}, max={max(cnt.values()) if cnt else 0}")
    if counts:
        print('Sınıf setleri eşit:', len({frozenset(c) for c in counts.values()}) == 1)
    overlaps = [v for v in hashes.values() if len({x[0] for x in v}) > 1]
    print(f"Farklı splitlerde birebir aynı dosya grubu: {len(overlaps)}")
    for overlap in overlaps[:20]:
        print('  ', ' | '.join(f"{sp}/{cls}/{Path(path).name}" for sp, cls, path in overlap))
    if len(overlaps) > 20:
        print(f"  ... {len(overlaps)-20} grup daha var")
    print("Not: yeniden boyutlandırılmış/yakın kopyalar SHA256 ile yakalanmaz.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="flower_data")
    main(Path(ap.parse_args().root))
