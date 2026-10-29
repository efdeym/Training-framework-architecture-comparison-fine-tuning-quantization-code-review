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