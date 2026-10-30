"""Measure CPU image-open + preprocess + model-forward latency (batch size 1).

From repository root:
  python scripts/benchmark_end_to_end.py --model flower_model_fp32.pt --images flower_data/valid --threads 4 --warmup 10 --iterations 100

Model-loading is intentionally excluded; use this alongside the original model-only benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

# Import deliver/predict.py without adding src/flowers as an extra requirement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deliver"))
from predict import FlowerClassifier  # noqa: E402


def find_images(root: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Image path not found: {root}")
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    if not paths:
        raise ValueError(f"No images found in {root}")
    random.Random(42).shuffle(paths)  # different classes/backgrounds for repeatable sampling
    return paths


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="One image or directory, e.g. flower_data/valid")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", default="results/end_to_end_benchmarks.csv")
    args = ap.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.iterations < 1:
        ap.error("Require threads>=1, warmup>=0, iterations>=1")

    torch.set_num_threads(args.threads)
    images = find_images(Path(args.images))
    classifier = FlowerClassifier(args.model, device="cpu")
    for i in range(args.warmup):
        classifier.predict(images[i % len(images)])

    times_ms = []
    for i in range(args.iterations):
        path = images[i % len(images)]
        start = perf_counter()
        classifier.predict(path)
        times_ms.append((perf_counter() - start) * 1000.0)

    data = {
        "model": args.model,
        "device": "cpu",
        "threads": args.threads,
        "images_available": len(images),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "batch_size": 1,
        "mean_ms": round(statistics.mean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(percentile(times_ms, 95), 3),
        "measurement": "open+preprocess+forward+argmax (excludes model initialization)",
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data))
        if new_file:
            writer.writeheader()
        writer.writerow(data)
    print(f"Saved measurement: {out}")


if __name__ == "__main__":
    main()
