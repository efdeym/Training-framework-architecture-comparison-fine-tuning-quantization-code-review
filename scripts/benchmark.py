"""KATMAN 7 - Giriş noktası. Amaç: TorchScript dosyasının ms/görüntü ve MB değerini ölçüp results/benchmarks.csv'ye eklemek.
Örnek: python scripts/benchmark.py --model flower_model.pt --device cpu --threads 1 --tag mnv3"""
import argparse, csv, json, os
import torch
from flowers.evaluation.benchmark import measure_latency_ms, model_size_mb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--tag", default="")
    ap.add_argument("--top1", type=float, default=float("nan"), help="aynı modelin doğruluğu (tabloya yazılır)")
    a = ap.parse_args()
    extra = {"meta.json": ""}
    m = torch.jit.load(a.model, map_location=a.device, _extra_files=extra)
    meta = json.loads(extra["meta.json"])
    dtype = torch.float16 if meta["dtype"] == "fp16" and a.device != "cpu" else torch.float32
    if dtype == torch.float32 and meta["dtype"] == "fp16":
        m = m.float()
    s = meta["img_size"]
    r = measure_latency_ms(m, (3, s, s), a.device, threads=a.threads, dtype=dtype)
    row = {"tag": a.tag or meta["arch"], "arch": meta["arch"], "device": a.device, "threads": a.threads,
           "img_size": s, "dtype": meta["dtype"], "size_mb": round(model_size_mb(a.model), 2),
           "mean_ms": round(r["mean_ms"], 3), "p95_ms": round(r["p95_ms"], 3), "top1": a.top1}
    print(row)
    os.makedirs("results", exist_ok=True)
    path = "results/benchmarks.csv"; new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); 
        if new: w.writeheader()
        w.writerow(row)
