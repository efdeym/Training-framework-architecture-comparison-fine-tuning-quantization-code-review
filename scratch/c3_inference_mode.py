"""C3 - torch.no_grad() ile torch.inference_mode() farki: hiz ve bellek (CPU).

Yontem:
- Kucuk bir evrisimli model, CPU, 1 thread, batch=1.
- Olcumden once her iki mod icin de warm-up yapilir (ilk calistirma etkisi elenir).
- Iki mod sirayla degil DONUSUMLU olculur; sira etkisini azaltmak icin her turda hangisinin once
  calisacagi degistirilir (A-B, B-A, A-B, ...).
- Ortalama, medyan ve standart sapma yazilir.
- Bellek icin psutil varsa RSS okunur; yoksa bellek olculmedi diye yazilir.
"""
import statistics
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)

N_WARMUP = 30   # her mod icin
N_ROUNDS = 300  # her turda iki mod da bir kez olculur

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
).eval()
x = torch.randn(1, 3, 128, 128)

MODES = {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode}

try:
    import os
    import psutil
    proc = psutil.Process(os.getpid())
    rss_mb = lambda: proc.memory_info().rss / 1024 / 1024
except ImportError:
    proc = None
    rss_mb = None


def one_call(ctx):
    with ctx():
        t0 = time.perf_counter()
        out = model(x)
        dt = (time.perf_counter() - t0) * 1000
    return dt, out


# --- Ciktinin durumu (deterministik gerceklikler) ---
print("torch:", torch.__version__, "| thread:", torch.get_num_threads())
print("\n[Cikti durumu]")
out_default = model(x)
print("varsayilan   : requires_grad =", out_default.requires_grad, "| grad_fn =", out_default.grad_fn is not None)
for name, ctx in MODES.items():
    with ctx():
        o = model(x)
    print(f"{name:14s}: requires_grad = {o.requires_grad} | grad_fn = {o.grad_fn is not None} | is_inference = {o.is_inference()}")

# --- Warm-up ---
for name, ctx in MODES.items():
    for _ in range(N_WARMUP):
        one_call(ctx)

# --- Donusumlu olcum ---
times = {k: [] for k in MODES}
mem = {k: [] for k in MODES}
names = list(MODES)
for r in range(N_ROUNDS):
    order = names if r % 2 == 0 else names[::-1]
    for name in order:
        dt, _ = one_call(MODES[name])
        times[name].append(dt)
        if rss_mb:
            mem[name].append(rss_mb())

print(f"\n[Sure, ms/cagri] warm-up={N_WARMUP}, tur={N_ROUNDS} (her mod {N_ROUNDS} olcum), donusumlu sira")
for name in names:
    t = times[name]
    print(f"{name:14s}: ortalama={statistics.mean(t):.3f}  medyan={statistics.median(t):.3f}  std={statistics.pstdev(t):.3f}  min={min(t):.3f}  max={max(t):.3f}")
mean_ng, mean_im = statistics.mean(times["no_grad"]), statistics.mean(times["inference_mode"])
med_ng, med_im = statistics.median(times["no_grad"]), statistics.median(times["inference_mode"])
print(f"fark (no_grad - inference_mode): ortalama={mean_ng - mean_im:+.3f} ms | medyan={med_ng - med_im:+.3f} ms")
print(f"std ile karsilastir: no_grad std={statistics.pstdev(times['no_grad']):.3f}, inference_mode std={statistics.pstdev(times['inference_mode']):.3f}")

print("\n[Bellek, RSS MB]")
if rss_mb:
    for name in names:
        m = mem[name]
        print(f"{name:14s}: min={min(m):.1f}  medyan={statistics.median(m):.1f}  max={max(m):.1f}")
    print("Not: RSS isletim sistemi duzeyinde, dalgali ve iki modun ayni surecte donusumlu calismasi nedeniyle ayristirmasi zayif bir olcudur.")
else:
    print("psutil kurulu degil, bellek OLCULMEDI.")
