"""C21/C22 - Dinamik ve statik INT8 kuantizasyon denemesi (eager model, CPU).

Notlar:
- results/benchmarks.csv'ye YAZMAZ; yalnizca terminale basar. Gecici dosyalar temizlenir.
- Tum varyantlar ayni eager yolda ve ayni olcum fonksiyonuyla (measure_latency_ms) olculur,
  boylece karsilastirma adildir. TorchScript (frozen) sayilariyla birebir kiyaslanmamalidir.
- Statik PTQ'da kalibrasyon verisi RASTGELE tensordur: yalnizca calisabilirlik testidir,
  gercek kalibrasyon degildir ve dogruluk olculmez.
Kullanim: python scratch/c21_c22_quant_experiment.py <ckpt_yolu>
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn

from flowers.compression.quantize import static_ptq_int8
from flowers.evaluation.benchmark import measure_latency_ms
from flowers.export.packager import load_checkpoint_model

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
args = parser.parse_args()

model, ck = load_checkpoint_model(args.ckpt)
size = ck["img_size"]
print("ckpt:", args.ckpt, "| arch:", ck["arch"], "| torch:", torch.__version__)
print("quantized engines:", torch.backends.quantized.supported_engines)


def state_mb(m):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        torch.save(m.state_dict(), p)
        return os.path.getsize(p) / 1024 / 1024


# Parametre dagilimi: Conv2d / Linear / diger
cnt = {"Conv2d": 0, "Linear": 0, "diger": 0}
seen = set()
for mod in model.modules():
    for p in mod.parameters(recurse=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        key = "Conv2d" if isinstance(mod, nn.Conv2d) else "Linear" if isinstance(mod, nn.Linear) else "diger"
        cnt[key] += p.numel()
total = sum(cnt.values())
print("parametre dagilimi:", {k: f"{v} ({100 * v / total:.1f}%)" for k, v in cnt.items()})


def run(name, m):
    r = measure_latency_ms(m, (3, size, size), "cpu", threads=1)
    print(f"{name:22s} state_dict_MB={state_mb(m):7.2f}  mean_ms={r['mean_ms']:7.3f}  p95_ms={r['p95_ms']:7.3f}")


run("eager fp32", model)

# C21: dinamik INT8 (yalnizca nn.Linear)
try:
    dyn = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    n_dyn = sum(1 for m in dyn.modules() if "dynamic" in type(m).__module__)
    print("dinamik kuantize edilen Linear sayisi:", n_dyn)
    run("dinamik int8 (Linear)", dyn)
except Exception as e:
    print("DINAMIK INT8 HATA:", type(e).__name__, e)

# C22: statik PTQ INT8 (rastgele kalibrasyon, yalnizca calisabilirlik)
try:
    calib = [(torch.randn(8, 3, size, size), torch.zeros(8)) for _ in range(5)]
    st = static_ptq_int8(model, calib, n_batches=5)
    run("statik int8 (rastgele)", st)
except Exception as e:
    print("STATIK INT8 HATA:", type(e).__name__, str(e)[:600])

