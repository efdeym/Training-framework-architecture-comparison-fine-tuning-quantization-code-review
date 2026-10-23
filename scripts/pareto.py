"""KATMAN 7 - Giriş noktası. Amaç: results/benchmarks.csv'den doğruluk - ms ve doğruluk - MB Pareto grafikleri çizmek."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmarks.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, col, lab in zip(ax, ["mean_ms", "size_mb"], ["ms / görüntü", "MB"]):
    a.scatter(df[col], df["top1"])
    for _, r in df.iterrows():
        a.annotate(r["tag"], (r[col], r["top1"]), fontsize=7)
    a.set_xlabel(lab); a.set_ylabel("top-1"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/pareto.png", dpi=140)
print("results/pareto.png yazıldı")
