# Ekip Bölüşümü ve Kurallar

| Kişi | Sorumluluk | Dokunduğu katmanlar |
|---|---|---|
| Lider | İskelet, protokol, kod incelemesi, Pareto raporu | 0, 1, 3, 4 |
| Kişi 2 | Hafif CNN'ler (MobileNetV3, EfficientNet, RegNet) | 2 (config'ler) |
| Kişi 3 | ViT/hibrit modeller + distillation öğretmeni | 2, 5 (distill) |
| Kişi 4 | Gölge test seti + kuantizasyon + teslim paketi | 5 (quantize), 6 |

## Altın kurallar
1. Aynı veri bölümü, aynı seed (42), aynı `scripts/benchmark.py`.
2. Her deney = bir YAML + `exp_name`. Kod içinde sabit sayı yok.
3. Karşılaştırma tablosu `results/benchmarks.csv`; elle sayı yazılmaz.
4. `final_fit: true` sadece model seçimi bittikten sonra, bir kez.
5. Kod incelemesi kontrol listesi: veri sızıntısı, normalizasyon, etiket eşlemesi, seed, `eval()` modu, benchmark adilliği.
