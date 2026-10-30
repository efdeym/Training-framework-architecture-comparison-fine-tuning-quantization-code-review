# Sonuçlar (yalnızca ölçülmüş değerler)

Ölçüm koşulları: `docs/PROTOCOL.md`. Doğruluk: `scripts/evaluate.py`, `best.pt` (mnv3_small_ft için `last.pt`, nedeni aşağıda). Valid 818, test 819 görsel.

## Doğruluk ve boyut (TorchScript fp32)
| Model | Eğitim | Valid | Test | MB (bayt÷1e6) | MiB (export.py) |
|---|---|---|---|---|---|
| mnv3_small_full | 10 epoch, backbone donuk | %93,03 | %92,31 | 6,50 | 6,20 |
| mnv3_small_ft | 30 epoch, LP-FT | %98,04 | %97,44 | 6,50 | 6,20 |
| mobilenetv3_large (mobilenetv3_full) | 30 epoch, LP-FT | %99,39 | %99,27 | 17,33 | 16,52 |
| efficientnet_b0 (efficientnet_b0_full) | 30 epoch, LP-FT | %99,14 | %99,02 | 16,53 | 15,76 |
| convnext_nano (convnext_nano_full) | 30 epoch, LP-FT + MixUp/CutMix | %99,39 | %99,39 | 60,10 | 57,32 |

## CPU gecikmesi (batch 1, fp32, yalnızca model çıkarımı, ortalama / p95 ms)
| Model | 1 thread | 14 thread (varsayılan) |
|---|---|---|
| mnv3_small_ft | 3,05 / 3,25 | 7,76 / 8,87 |
| mobilenetv3_large | 8,39 / 9,44 | 15,64 / 18,99 |
| efficientnet_b0 | 16,40 / 17,82 | 19,53 / 26,07 |
| convnext_nano | 52,47 / 55,61 | ölçülmedi |

Batch=1'de 14 thread, 1 thread'den yavaş çıktı. Ön işleme dahil uçtan uca süre henüz ölçülmedi. Tek ölçüm, ~%5 gürültü var.

## Test anında giriş boyutu (mobilenetv3_large, yeniden eğitmeden, yalnızca valid)
224: %99,39, 256: %99,27, 288: %99,14, 320: %98,53. Büyütme kazandırmadı.

## Notlar
- mnv3_small_ft: `best.pt` 28.10'da aynı exp_name ile başlatılıp yarıda kalan bir çalıştırmayla ezildi; `last.pt` (30. epoch) kullanıldı. Log'daki en iyi valid %98,53 (epoch 28), ağırlıkları kayıp.
- ConvNeXt-nano 35 MB ve 50 ms sınırını aşıyor, elendi.
- Gölge test, uçtan uca süre, TTA/ensemble: bekliyor. Nihai model kararı henüz verilmedi.

### EffNet-B0 eğri notu (A19)
Valid 9-13. epoch'ta doygunluğa ulaştı (%98,9-99,1); sonraki 17 epoch iyileşme getirmedi. 
Large ile fark 2 görsel (gürültü sınırı); hız farkı (16,4 vs 8,4 ms) belirleyici. 
Aday: Large. Not: train.log test_top1=0,9927 (last.pt?) ile evaluate.py test=0,9902 (best.pt?) farkı kontrol edilecek.

### LP-FT deneyi (A21)
LP=3 (ana koşu) valid %99,27 / en iyi %99,39; LP=0 valid %99,14 / en iyi %99,39. 
Fark 1 görsel (gürültü sınırı). Karar: LP=3 varsayılan kalır. LP=0 daha hızlı ısınıyor ama sonuç aynı.

### Aday seçimi (A26)
Kural: doğruluk gürültü sınırı içinde eşitse en düşük ms ve MB seçilir.
- ConvNeXt-nano: doğruluk Large ile eşit (%99,39), ama 60 MB ve 52 ms; elendi.
- EffNet-B0: Large'dan 2 görsel geride, 2 kat yavaş; elendi.
- Large: valid %99,39 / test %99,27, yeni export 8,62 ms (p95 9,10), 16,52 MB.
Gölge test doğrulanmadığı için karar dışı bırakıldı; yerine valid üzerinde bozulma testi yapıldı (Large en fazla %1,1 düştü).
Karar: mobilenetv3_large_100, LP-FT (LP=3), EMA 0,999, 224. Doğruluk gizli testte %99 ise tahmini skor ≈ 99,4 (hız ve boyut tavanda).

### Ön işleme ve bozulma dayanıklılığı (valid, 818 görsel)
Varsayılan %99,39; 224'e sıkıştırma %98,90; bulanıklık %98,29; karanlık %98,90; parlak %98,41; JPEG-20 %98,66. Ön işleme farkı en fazla 4 görsel; bozulmalarda düşüş en fazla %1,1. Script: scripts/robustness_probe.py.

### Temiz ortam testi
timm kurulu olmayan ortamda (torch 2.5.1+cpu, numpy, pillow) deliver/predict.py flower modelini yükleyip tahmin üretti.

### Uçtan uca süre (final model)
JPEG çözme + resize + model (predict.py), 1 thread, 200 test görseli: 35,28 ms/görsel (yalnızca model ~9,1 ms). 50 ms sınırının altında, pay 1,4 kat; ön işleme dökümü için scripts/e2e_profile.py.

### Son kod incelemesi (A33)
- Veri sızıntısı: image_08077 dışlandı, MD5 taraması train/test ve valid/test 0 çakışma
- Normalizasyon: modele gömülü, checkpoint ile TorchScript 10/10 aynı sınıf (max logit farkı 4,5e-6)
- Etiket eşlemesi: meta.json ve checkpoint eşlemesi aynı (0→"1", 1→"10", 2→"100")
- Seed: seed_everything(42) train.py'de çağrılıyor; cudnn.benchmark açık, GPU eğitimi bit-bit tekrarlanabilir değil
- model.eval(): predict, benchmark ve evaluate yolunda doğrulandı
- Benchmark adilliği: 20 ısınma, 200 ölçüm, batch=1, inference_mode, 1 thread

### Vazgeçilen maddeler
A23: 192/160 ile yeniden eğitim yapılmadı; hız ve boyut tavanda, küçültmenin kazancı yok, doğruluk riski var.
A25: distillation yapılmadı; doğruluk %99+ ve hız/boyut tavanda.