# Flower Classifier (102 sınıf)

Çiçek türü sınıflandırıcısı. Değerlendirme kriterleri: **top-1 doğruluk (gizli test)**, **ms/görüntü**, **MB**.
Katmanlı mimari ve sınıf amaçları için: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Ekip planı: [`docs/TEAM.md`](docs/TEAM.md).

## Kurulum
```bash
pip install -r requirements.txt
pip install -e .          # veya: export PYTHONPATH=src
```

## Veri
```
data/flowers/
  train/<sınıf_no>/*.jpg    (6.552 görüntü)
  valid/<sınıf_no>/*.jpg    (818)
  test/<sınıf_no>/*.jpg     (921)
```
Klasör adları sayısal metindir (1, 10, 100...). `ImageFolder` bunları alfabetik indeksler; eşleme checkpoint'e ve teslim dosyasına
otomatik kaydedilir, `predict` **orijinal klasör adını** döndürür.

## Hızlı başlangıç
```bash
# 1) Baseline eğitim
python scripts/train.py --config configs/mobilenetv3.yaml

# 2) Başka mimari / override
python scripts/train.py --config configs/convnext_nano.yaml train.epochs=40 exp_name=cnxt_a

# 3) Gölge test (farklı kaynaktan topladığın görüntüler, aynı klasör adlarıyla) üzerinde değerlendir
python scripts/evaluate.py --ckpt runs/mnv3_large/best.pt --data data/shadow_test

# 4) Teslim dosyası üret (tek dosya, timm gerekmez)
python scripts/export.py --ckpt runs/mnv3_large/best.pt --out flower_model.pt --dtype fp32

# 5) Hız + boyut ölç ve tabloya ekle
python scripts/benchmark.py --model flower_model.pt --device cpu --threads 1 --tag mnv3 --top1 0.97
python scripts/pareto.py       # results/pareto.png
```

## Önerilen deney sırası
1. `mobilenetv3`, `efficientnet_b0`, `convnext_nano` ile baseline (aynı recipe)  ->  Pareto
2. Çözünürlük (160/192/224) ve recipe (LP-FT, EMA, MixUp/CutMix) denemeleri
3. Öğretmen eğit (`convnext_base...`, `runs/teacher/`) -> `configs/distill_student.yaml`
4. fp16 / int8 sıkıştırma (`flowers.compression.quantize`), her adımda doğruluk ölç
5. **Son:** `configs/final_fit.yaml` (train+valid+test birleşik), `last.pt` ile export

## Teslim: modeli kullanma (değerlendiren kişi için)
Gereken: `flower_model.pt` ve `predict.py` (sadece `torch`, `numpy`, `Pillow`).
```python
from predict import FlowerClassifier

clf = FlowerClassifier("flower_model.pt", device="cpu")   # GPU: device="cuda"
print(clf.predict("foto.jpg"))                            # örn. "73" (orijinal sınıf adı)
print(clf.predict_batch(["a.jpg", "b.jpg"]))
```
Doğrudan PyTorch ile (kendi ön işlemenle):
```python
import json, torch
extra = {"meta.json": ""}
model = torch.jit.load("flower_model.pt", _extra_files=extra).eval()
meta = json.loads(extra["meta.json"])      # idx_to_class, mean, std, img_size, crop_pct
# giriş: [B,3,img_size,img_size], RGB, [0,1] -> (x-mean)/std ; çıkış: logits [B,102]
# sınıf adı: meta["idx_to_class"][str(idx)]
```
> Not: `torch.jit` yeni PyTorch sürümlerinde "deprecated" uyarısı verir, işlev görür. Değerlendirme ortamının
> PyTorch sürümüyle aynı/yakın sürümde export edip test et.

## Bilinen riskler ve notlar
- **Domain shift:** Gizli test bu veri setinden değil. Val/test %99 olsa da gölge testle doğrula.
- **Hue augmentation yok:** Renk çiçek türünü belirler.
- **fp16:** GPU'da hızlı/küçük; CPU'da `predict.py` otomatik fp32'ye çevirir (dosya boyutu yine yarı). fp16 TorchScript yolunu hedef donanımda ayrıca doğrula.
- **int8 (FX PTQ):** CPU hedefliyorsan dene; model bazında sorun çıkarabilir, her zaman doğruluğu yeniden ölç.
- **Hız ölçümü:** CPU/GPU belirsiz. `benchmark.py` ikisini de destekler; ön işleme dahil süre için `evaluation.benchmark.measure_end_to_end_ms`.
- **Pretrained ağırlıklar** timm üzerinden internetten (Hugging Face Hub) iner; ilk çalıştırmada bağlantı gerekir.

## Test
```bash
pytest -q tests
```
Testler: etiket eşleme, `predict.py` ön işlemesinin eğitim eval transform'una uyumu, LR scheduler.

## Doğrulama durumu
Sahte 4 sınıflı veriyle `train -> export -> predict -> evaluate -> benchmark -> pareto` zinciri uçtan uca çalıştırıldı (CPU).
Gerçek veri ve pretrained ağırlıkla eğitim henüz yapılmadı.
=======
# Training-framework-architecture-comparison-fine-tuning-quantization-code-review
>>>>>>> 672f4091a794cd4182c1608a9baf7682fdc88d0a
