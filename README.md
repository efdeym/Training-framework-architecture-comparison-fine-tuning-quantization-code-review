# Flower Classifier (102 sınıf)

Çiçek türü sınıflandırıcısı. Değerlendirme kriterleri: **top-1 doğruluk (gizli test)**, **ms/görüntü**, **MB**.
Hocanın önceliği: **en kısa sürede çalışan ve en az yer kaplayan model**. Bu yüzden teslim adayı olarak MobileNet ailesine odaklanıyoruz.

Katmanlı mimari ve sınıf amaçları için: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Ekip planı ve Git kuralları: [`docs/TEAM.md`](docs/TEAM.md).

## Durum

| Aşama | Durum |
|---|---|
| Kurulum, testler | Tamam |
| Gerçek veride eğitim ve adil kıyas (Small, Large, EffNet-B0, ConvNeXt-nano) | Tamam |
| LP-FT, LP epoch sayısı, test anı giriş boyutu, kuantizasyon kararı | Tamam |
| Export, normalizasyon modele gömme, temiz ortam testi | Tamam |
| EMA (0,999) | Valid %99,27 / test %99,51 (log); baseline %99,27 / %99,39, fark 1 görsel; eğri daha düzgün | Maliyetsiz, final_fit'te kullanılabilir (A27) |
| Nihai `flower_model.pt`, uçtan uca süre ölçümü | Sırada |
| Gölge test | Doğrulanmadı, bkz. Bilinen riskler |
| Organizatör soruları | Cevap alınamadı, varsayımlar aşağıda |

## Sonuçlar


Yalnızca ölçülmüş değerler. Doğruluklar `evaluate.py` ile (valid 818, test 819 görsel); gecikme CPU, batch=1, yalnızca model ileri geçişi.

| Model | Eğitim | Valid | Test | CPU ms (1 thr / 14 thr) | MB |
|---|---|---|---|---|---|
| `mobilenetv3_small_100` | 10 epoch, backbone donuk | %93,03 | %92,31 | 3,2 / - | 6,50 |
| `mobilenetv3_small_100` | 30 epoch, LP-FT | %98,04 | %97,44 | 3,05 / 7,76 | 6,50 |
| **`mobilenetv3_large_100`** | 30 epoch, LP-FT | **%99,39** | **%99,27** | **8,39** / 15,64 | 17,33 |
| `efficientnet_b0` | 30 epoch, LP-FT | %99,14 | %99,02 | 16,40 / 19,53 | 16,53 |
| `convnext_nano` | 30 epoch, LP-FT | %99,39 | %99,39 | 52,47 / - | 60,10 |

**Nasıl okunmalı**
- Valid'de 1 görsel = %0,12. Large ile EffNet arasındaki 2 görsellik fark gürültü sınırındadır; hız farkı (8,4 vs 16,4 ms) belirleyicidir.
- ConvNeXt boyuttan (60 MB) ve hızdan elendi.
- Önerilen sınırlar (50 ms, 35 MB) kesinleşmedi; Large bunların çok altında kalıyor.
- Gecikmeler geliştirme bilgisayarına aittir. Batch=1'de çok thread, tek thread'den yavaştır.
- Valid/test bu veri setindendir; gizli test için mutlak başarıyı değil, modeller arası kıyası gösterir.

### Optimizasyon deneyleri (yaramayanlar dahil)

| Deney | Sonuç | Karar |
|---|---|---|
| Backbone donuk vs LP-FT (Small) | %93,0 → %98,0 | LP-FT kullanılır |
| LP epoch 3 vs 0 (Large) | %99,27 vs %99,14, fark 1 görsel | LP=3 kalır |
| Test anında giriş büyütme (256/288/320) | Doğruluk düştü | 224'te kalındı |
| Çok thread (batch=1) | 15,6 ms vs 8,4 ms (1 thread) | Ek ayar yapılmadı |
| fp16 / int8 kuantizasyon | Hız ve boyut zaten sınırların altında | Atlandı |
| TTA (flip vb.) | Ölçülmedi, süreyi iki katlar | Eklenmedi |
| Large + EffNet ensemble | 33,9 MB, 35 MB sınırına 1,1 MB | Bırakıldı (risk) |
| EMA (0,999) | Sonuç bekleniyor | - |

### Ön işleme dayanıklılığı (valid, 818 görsel, Large)

| Varyant | Doğruluk |
|---|---|
| Varsayılan (resize + merkez kırp) | %99,39 |
| 224x224'e sıkıştırma | %98,90 |
| Bulanıklık (sigma=2) | %98,29 |
| Karanlık (x0,6) | %98,90 |
| Parlak (x1,4) | %98,41 |
| JPEG kalite 20 | %98,66 |

Script: `scripts/robustness_probe.py`.

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

**final_fit kararı (A27):** Seçilen reçete (Large, LP-FT, EMA 0,999, 224, 30 epoch, seed 42) train+valid+test birleşimiyle bir kez yeniden eğitilir (+%25 veri). Bu eğitimden sonra valid/test'te ölçülen doğruluk anlamsızdır (veri eğitimde görüldü); yalnızca sağlık kontrolü yapılır (~%100 beklenir). Yedek: `runs/mnv3_large_ema` ve `runs/mobilenetv3_full`. Kazanç küçüktür (gizli testte birkaç görsel), riski sağlık kontrolü ve yedekle sınırlandırıldı.

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
meta = json.loads(extra["meta.json"])      # idx_to_class, img_size, crop_pct, norm_in_model
# giriş: [B,3,img_size,img_size], RGB, [0,1] aralığında. Normalizasyon (mean/std) MODELİN İÇİNDE, dışarıda uygulama.
# ön işleme: kısa kenarı int(img_size/crop_pct) yap (bicubic), merkez kırp img_size x img_size
# çıkış: logits [B,102]; sınıf adı: meta["idx_to_class"][str(idx)]
```

> Not: Model, hem ham [0,1] aralığındaki RGB tensörleri hem de ImageNet ile önceden normalize edilmiş girdileri otomatik algılar ve sorunsuz kabul eder (çifte normalizasyona karşı korumalıdır).

> Not: `torch.jit` yeni PyTorch sürümlerinde "deprecated" uyarısı verir, işlev görür. Değerlendirme ortamının PyTorch sürümüyle aynı veya yakın sürümde export edip test et.


## Değerlendirme koşulları (organizatörden cevap bekleniyor)

Rehberde yer alanlar: Intel i7, 16 GB, GPU yok; skor = 0,60 x doğruluk + 0,25 x hız + 0,15 x boyut; Tmax=50 ms, Smax=35 MB (önerilen). Şu varsayımlarla ilerliyoruz:

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