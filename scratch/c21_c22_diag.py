"""C21/C22 tani - dinamik INT8 neden 0 Linear donusturdu, statik PTQ onednn ile calisir mi?
Yalnizca terminale basar; dosya yazmaz. Kullanim: python scratch/c21_c22_diag.py <ckpt>
"""
import sys
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
from flowers.compression.quantize import static_ptq_int8
from flowers.evaluation.benchmark import measure_latency_ms
from flowers.export.packager import load_checkpoint_model

model, ck = load_checkpoint_model(sys.argv[1])
lin = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
print("Linear katmanlari:", lin)
print("classifier tipi:", type(model.get_classifier()).__name__)

print("engine (varsayilan):", torch.backends.quantized.engine, "| desteklenen:", torch.backends.quantized.supported_engines)
dyn = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
print("dyn classifier tipi:", type(dyn.get_classifier()).__module__, type(dyn.get_classifier()).__name__)
print("dyn tum Linear-benzeri tipler:", sorted({type(m).__module__ + "." + type(m).__name__ for _, m in dyn.named_modules() if "inear" in type(m).__name__}))

size = ck["img_size"]
calib = [(torch.randn(8, 3, size, size), torch.zeros(8)) for _ in range(5)]
try:
    st = static_ptq_int8(model, calib, backend="onednn", n_batches=5)
    with torch.inference_mode():
        out = st(torch.randn(1, 3, size, size))
    print("statik onednn: calisti, cikti shape", tuple(out.shape))
    r = measure_latency_ms(st, (3, size, size), "cpu", threads=1)
    print(f"statik onednn (rastgele kalibrasyon): mean_ms={r['mean_ms']:.3f} p95_ms={r['p95_ms']:.3f}")
except Exception as e:
    print("statik onednn HATA:", type(e).__name__, str(e)[:600])

