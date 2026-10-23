# Mimari: 8 Katman ve Sınıf/Modül Amaçları

Bağımlılık yönü **yukarıdan aşağıya**: üst katman alt katmanı kullanır, tersi olmaz.

```
KATMAN 7  scripts/            (giriş noktaları / CLI)
KATMAN 6  export/ + deliver/  (teslim paketi)
KATMAN 5  compression/        (distillation, kuantizasyon)
KATMAN 4  evaluation/         (metrik, hız, boyut)
KATMAN 3  training/           (eğitim döngüsü)
KATMAN 2  models/             (mimari, EMA)
KATMAN 1  data/               (veri, transform, etiket)
KATMAN 0  config + utils/     (altyapı)
```

## KATMAN 0 - Altyapı
| Dosya | Amaç |
|---|---|
| `config.py` (`Config`, `load_config`) | Tüm ayarlar tek şemada. YAML + `a.b=c` override. Her config `base.yaml`'ı miras alır. |
| `utils/seed.py` (`seed_everything`) | Tekrarlanabilirlik. Ekipte herkes aynı seed. |
| `utils/logging.py` (`get_logger`, `MetricLogger`) | Dosya/konsol log + opsiyonel TensorBoard/W&B. |

## KATMAN 1 - Veri
| Dosya | Amaç |
|---|---|
| `data/labels.py` | **idx <-> klasör adı** eşlemesi. ('1','10','100' alfabetik sıralanır; idx != klasör no.) |
| `data/transforms.py` | Eğitim (RandomResizedCrop, flip, hafif jitter, **hue yok**) ve eval (resize->center crop->normalize) dönüşümleri. Eval = `deliver/predict.py` ile aynı olmalı. |
| `data/datasets.py` | `ImageFolder`, split tutarlılık kontrolü, karekök-ağırlıklı sampler, `final_fit` ile train+valid+test birleştirme. |

## KATMAN 2 - Model
| Dosya | Amaç |
|---|---|
| `models/factory.py` | `create_model(name)` (timm), pretrained mean/std okuma, backbone/kafa parametre ayrımı. Mimari karşılaştırması = sadece `model.name` değişir. |
| `models/ema.py` (`ModelEma`) | Üstel hareketli ortalama ağırlıklar. |

## KATMAN 3 - Eğitim
| Dosya | Amaç |
|---|---|
| `training/losses.py` | Label smoothing CE, MixUp/CutMix için soft-target CE. |
| `training/optim.py` | AdamW (kafa için daha yüksek LR) + warmup'lı cosine (adım bazlı). |
| `training/engine.py` | `train_one_epoch` (AMP, grad clip, EMA, mixup, KD) ve `predict_loader`. |
| `training/trainer.py` (`Trainer`) | Hepsini birleştirir: LP-FT, checkpoint (+ class_to_idx, mean/std), loglama. |

## KATMAN 4 - Değerlendirme
| Dosya | Amaç |
|---|---|
| `evaluation/metrics.py` | top-1, sınıf bazlı doğruluk, confusion matrix. |
| `evaluation/benchmark.py` | ms/görüntü (warmup, batch=1, sabit thread, CUDA sync), MB, uçtan uca (ön işleme dahil) süre. |

## KATMAN 5 - Sıkıştırma
| Dosya | Amaç |
|---|---|
| `compression/distill.py` | Öğretmen yükleme + KD kaybı (T, alpha). |
| `compression/quantize.py` | fp16 ve FX statik PTQ int8 (CPU). Deneysel; doğruluğu yeniden ölç. |

## KATMAN 6 - Teslim
| Dosya | Amaç |
|---|---|
| `export/packager.py` | Checkpoint -> tek dosya TorchScript; etiket eşlemesi + ön işleme ayarları `meta.json` olarak içine gömülür. |
| `deliver/predict.py` (`FlowerClassifier`) | Bağımsız tahmin sınıfı. Sadece torch+numpy+Pillow. Orijinal sınıf adını döndürür. |

## KATMAN 7 - Giriş noktaları
| Script | Amaç |
|---|---|
| `scripts/train.py` | Config ile eğitim. |
| `scripts/evaluate.py` | Checkpoint'i herhangi bir klasörde (ör. gölge test) değerlendir. |
| `scripts/export.py` | Teslim dosyasını üret (fp32/fp16). |
| `scripts/benchmark.py` | ms ve MB ölç, `results/benchmarks.csv`'ye ekle. |
| `scripts/pareto.py` | Doğruluk-ms ve doğruluk-MB grafiği. |
| `scripts/make_fake_data.py` | Pipeline duman testi için sahte veri. |
