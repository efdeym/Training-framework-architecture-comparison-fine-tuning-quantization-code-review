# Flower Classifier (102 sınıf)

Çiçek türü sınıflandırıcısı. Değerlendirme kriterleri: **top-1 doğruluk (gizli test)**, **ms/görüntü**, **MB**.
Hocanın önceliği: **en kısa sürede çalışan ve en az yer kaplayan model**. Bu yüzden teslim adayı olarak MobileNet ailesine odaklanıyoruz.

Katmanlı mimari ve sınıf amaçları için: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Ekip planı ve Git kuralları: [`docs/TEAM.md`](docs/TEAM.md).

## Durum

| Aşama | Durum |
|---|---|
| Kurulum, testler (3/3 geçiyor) | Tamam |
| Sahte veriyle uçtan uca duman testi | Tamam |
| Gerçek veride eğitim (MobileNetV3-Large, MobileNetV3-Small) | Tamam |
| Boyut ve gecikme ölçümü (Small) | Tamam |
| Large ile adil kıyas (aynı epoch, aynı ayar) | Sırada |
| fp16 / int8 sıkıştırma, giriş boyutu denemeleri | Sırada |
| Gölge test seti (farklı kaynaklı görseller) | Sırada |
| Organizatör soruları (donanım, ölçüm biçimi, teslim formatı) | Cevap bekleniyor |

## Sonuçlar

Yalnızca ölçülmüş değerler yazılmıştır. Boş bırakılan hücreler henüz ölçülmedi.

| Model | Eğitim | Valid top-1 | Test top-1 | Gecikme ort. (ms) | Dosya (MB) |
|---|---|---|---|---|---|
| `mobilenetv3_small_100` | 10 epoch, backbone donuk (yalnızca kafa) | %93,0 (en iyi, epoch 8) | %91,8 | 3,02 | 6,20 |
| `mobilenetv3_large_100` | 1 epoch, backbone donuk | %88,6 | %87,5 | ölçülmedi | ölçülmedi |

**Nasıl okunmalı**
- İki satır **adil bir kıyas değildir**: Small 10 epoch, Large yalnızca 1 epoch eğitildi. Large'ı aynı ayarla (10 epoch, backbone donuk) tekrar eğitip ölçtükten sonra karşılaştırma yapılacak.
- Gecikme koşulu: CPU, 1 thread, batch=1, 224x224 giriş, yalnızca modelin ileri geçişi (görsel okuma, yeniden boyutlandırma ve normalizasyon dahil değil). Ek bilgi: p95 gecikme 3,52 ms. Bu değerler geliştirme bilgisayarına aittir; değerlendirme makinesinde farklı çıkar.
- Valid/test doğrulukları bu veri setindendir. Gizli test başka kaynaktan geleceği için bu sayılar mutlak başarıyı değil, yalnızca modeller arası kıyası gösterir.

## Kurulum

```bash
python -m venv .venv
# Windows (cmd):         .venv\Scripts\activate.bat
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Mac/Linux:             source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # veya: PYTHONPATH=src
pytest -q tests           # 3 test geçmeli
```

Python 3.10 veya üstü gerekir.

## Veri

```
data/flowers/
  train/<sınıf_no>/*.jpg    (6.552 görüntü)
  valid/<sınıf_no>/*.jpg    (818)
  test/<sınıf_no>/*.jpg     (819)
```

102 sınıf, üç bölümde de aynı sınıflar. Train'de sınıf başına 27-206 görüntü var (dengesiz), valid'te sınıf başına 1-28 görüntü var, bu yüzden valid doğruluğu biraz oynaktır.

Klasör adları sayısal metindir (1, 10, 100...). `ImageFolder` bunları alfabetik indeksler; eşleme checkpoint'e ve teslim dosyasına otomatik kaydedilir, `predict` **orijinal klasör adını** döndürür.

Veri arşivden çıkarma (Windows):

```bash
mkdir data\flowers
tar -xzf flower_data.tar.gz -C data\flowers
```

`data/` ve `*.tar.gz` Git'e girmez (`.gitignore`).

## Hızlı başlangıç

```bash
# 1) Baseline eğitim (MobileNetV3-Small, backbone donuk, 10 epoch)
python scripts/train.py --config configs/mnv3_small.yaml data.root=data/flowers

# 2) Başka mimari / override
python scripts/train.py --config configs/mobilenetv3.yaml data.root=data/flowers train.epochs=10 train.linear_probe_epochs=10 exp_name=mnv3_large_lp10

# 3) Teslim dosyası üret (tek dosya, timm gerekmez)
python scripts/export.py --ckpt runs/mnv3_small/best.pt --out mnv3_small.pt --dtype fp32

# 4) Hız + boyut ölç ve tabloya ekle (--top1 yerine eğitimin verdiği gerçek değeri yaz)
python scripts/benchmark.py --model mnv3_small.pt --device cpu --threads 1 --tag mnv3_small --top1 0.9303
python scripts/pareto.py       # results/pareto.png

# 5) Gölge test (farklı kaynaktan topladığımız görüntüler, aynı klasör adlarıyla) üzerinde değerlendir
python scripts/evaluate.py --ckpt runs/mnv3_small/best.pt --data data/shadow_test
```

## Deney sırası

1. `mnv3_small` ve `mobilenetv3` (Large) ile baseline, aynı ayarlarla, Pareto grafiği
2. Hoca küçük ve hızlı model istediği için küçük adaylar: `mobilenetv3_small_050`, `mobilenetv2_100`, `mobilenetv4_conv_small`
3. Çözünürlük (160/192/224) ve recipe (LP-FT, EMA) denemeleri
4. fp16 / int8 sıkıştırma (`flowers.compression.quantize`), her adımda doğruluk ve MB ölçümü
5. Opsiyonel: büyük bir öğretmenle (ör. ResNet50) distillation (`configs/distill_student.yaml`)
6. **Son:** `configs/final_fit.yaml` (train+valid+test birleşik), `last.pt` ile export

**ResNet50 hakkında:** Yaklaşık 95-100 MB olduğu için teslim modeli olarak uygun değil. Karşılaştırma referansı veya distillation öğretmeni olarak kullanılabilir.

**Seçim kuralı (öneri):** Gölge testte ekibin belirlediği doğruluk tabanının altına düşmeyen modeller arasından en küçük MB ve en düşük ms'li olan seçilir.

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

> Not: `torch.jit` yeni PyTorch sürümlerinde "deprecated" uyarısı verir, işlev görür. Değerlendirme ortamının PyTorch sürümüyle aynı veya yakın sürümde export edip test et.

## Değerlendirme koşulları (organizatörden cevap bekleniyor)

Cevap gelene kadar şu varsayımlarla ilerliyoruz; cevap farklı çıkarsa yalnızca ölçüm ayarları değişir:

- Donanım: CPU, 1 thread
- Ölçüm: batch=1, yalnızca modelin ileri geçişi
- Teslim: tek dosya TorchScript + bağımsız `predict.py`
- Tahmin çıktısı: orijinal sınıf klasör adı

Sorulan konular: CPU/GPU, ön işleme süreye dahil mi, teslim formatı ve PyTorch sürümü, girdi/çıktı biçimi, puanlama ağırlıkları, önceden eğitilmiş ağırlık ve dış veri kullanımı.

## Bilinen riskler ve notlar

- **Domain shift:** Gizli test bu veri setinden değil. Bu veri setindeki yüksek doğruluk yeni fotoğraflarda aynı kalmayabilir; gölge testle doğrulanacak.
- **Hue augmentation yok:** Renk çiçek türünü belirler.
- **fp16:** GPU'da hızlı ve yarı boyutta; CPU'da `predict.py` otomatik fp32'ye çevirir (dosya boyutu yine yarı). fp16 TorchScript yolunu hedef donanımda ayrıca doğrula.
- **int8 (FX PTQ):** CPU hedefliyorsak dene; model bazında sorun çıkarabilir, her zaman doğruluğu yeniden ölç.
- **Hız ölçümü:** CPU/GPU belirsiz. `benchmark.py` ikisini de destekler; ön işleme dahil süre için `evaluation.benchmark.measure_end_to_end_ms`.
- **Pretrained ağırlıklar** timm üzerinden internetten (Hugging Face Hub) iner; ilk çalıştırmada bağlantı gerekir. Windows'ta çıkan "symlinks" uyarısı zararsızdır.
- **final_fit:** Train+valid+test birleşik eğitimden sonra bu veride ölçülen doğruluk anlamsızdır; karar gölge testle verilir.

## Test

```bash
pytest -q tests
```

Testler: etiket eşleme, `predict.py` ön işlemesinin eğitim eval transform'una uyumu, LR scheduler.