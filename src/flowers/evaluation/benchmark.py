"""KATMAN 4 - Değerlendirme. Amaç: değerlendirme kriterlerinin ikisini ölçmek: ms/görüntü ve MB.
Adil ölçüm: warmup, batch=1, inference_mode, sabit thread, CUDA'da synchronize."""
import os, time, statistics
import torch


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


@torch.inference_mode()
def measure_latency_ms(model, input_size=(3, 224, 224), device="cpu", n_warmup=20, n_runs=200,
                       threads: int | None = 1, channels_last=False, dtype=torch.float32) -> dict:
    if threads:
        torch.set_num_threads(threads)
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(1, *input_size, device=dev, dtype=dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(n_warmup):
        model(x)
    sync()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter(); model(x); sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts),
            "p95_ms": sorted(ts)[int(0.95 * len(ts))], "std_ms": statistics.pstdev(ts)}


def measure_end_to_end_ms(predict_fn, image_paths, n_warmup=5) -> float:
    """Ön işleme (decode+resize) DAHİL ortalama süre. predict_fn(path) -> label."""
    for p in image_paths[:n_warmup]:
        predict_fn(p)
    t0 = time.perf_counter()
    for p in image_paths:
        predict_fn(p)
    return (time.perf_counter() - t0) * 1000 / len(image_paths)
