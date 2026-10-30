# C notları

## C1 - PyTorch temelleri: tensor, shape, dtype, device

Deneme scripti: `scratch/c1_tensor_basics.py` (torch 2.14.1+cpu, GPU yok).

1. Tensor, sayıların çok boyutlu dizisidir. `torch.arange(12)` 12 elemanlı, tek boyutlu bir tensor verir.
2. `shape` boyutları söyler. `reshape(3, 4)` veriyi değiştirmez, yalnızca nasıl bölündüğünü değiştirir.
3. Görsel modele `[batch, kanal, yükseklik, genişlik]` biçiminde girer. Örnek: `(1, 3, 224, 224)`. Tek görsel için batch=1.
4. `dtype` sayı tipidir. Tamsayı tensor `float()` ile fp32, `half()` ile fp16 olur. Eleman başına bayt: int64 = 8, fp32 = 4, fp16 = 2. fp16 modeli yaklaşık yarı boyuta indirmesi beklenir, C20'de ölçeceğiz.
5. `device` tensor'un yaşadığı yerdir (`cpu` veya `cuda`). Bu makinede `cuda` yok, her şey `cpu`'da.
6. İki tensor aynı cihazda olmalı. Farklı dtype'lar karışırsa PyTorch tipi yükseltir (fp32 + fp64 = fp64). Shape uyumsuzsa `RuntimeError` verir. fp32 ağırlıklı modele fp16 girdi vermek hata verir, model ve girdi aynı dtype'ta olmalı; bu C20 ve predict.py için önemli. Denendi (`scratch/c1b_dtype_mismatch.py`, fp32 `nn.Linear` + fp16 girdi): `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`. Girdiyi `.float()` yapınca hata gitti. (Yalnızca `nn.Linear` ile denendi, evrişimli model ve TorchScript ile denemedik.)
7. Normalize `(x - mean) / std` biçimindedir. `mean` ve `std` `(1, 3, 1, 1)` biçiminde verilince her kanala ayrı uygulanır (broadcasting).
8. Bu proje için önemi: model girdisi `[B, 3, img_size, img_size]`, RGB, `[0,1]` aralığında ve mean/std ile normalize olmalı (`deliver/predict.py` bunu yapar).

## C2 - model.train() ile model.eval() farkı

Deneme scripti: `scratch/c2_train_eval.py`.

`model.train()` modunda Dropout rastgele nöronları kapatır ve BatchNorm o anki batch'in ortalama/varyansını kullanıp `running_mean` değerini günceller. `model.eval()` modunda Dropout kapanır ve BatchNorm kayıtlı `running_mean`/`running_var` değerlerini kullanır, yani aynı girdi her seferinde aynı çıktıyı verir. Denemede aynı girdiyle `train()` içinde iki çağrı farklı, `eval()` içinde iki çağrı aynı çıktıyı verdi ve `running_mean` yalnızca train çağrısında değişti.

## C3 - torch.no_grad() ile torch.inference_mode() farkı

Deneme scripti: `scratch/c3_inference_mode.py` (CPU, 1 thread, torch 2.14.1+cpu, küçük evrişimli model, girdi 1x3x128x128).

`no_grad` ile `inference_mode` ikisi de gradyan hesabını kapatır, çıktıda `requires_grad=False` ve `grad_fn` yok. Denemede ölçülen fark yalnızca şuydu: `inference_mode` çıktısı `is_inference() = True`, `no_grad` çıktısı `False`.

Süre ölçümü (warm-up 30, dönüşümlü sıra, her mod için 300 ölçüm, ms/çağrı):

| Mod | Ortalama | Medyan | Std |
|---|---|---|---|
| no_grad | 7.071 | 6.920 | 1.261 |
| inference_mode | 7.066 | 6.880 | 1.227 |

Ortalama fark 0.005 ms, medyan farkı 0.040 ms. Her iki mod için standart sapma yaklaşık 1.2 ms olduğundan bu fark gürültü içinde, bu deneyde iki mod arasında hız farkı gözlenmedi. Bu sonuç tek bir model, tek bir makine ve tek bir çalıştırma içindir. Zamanlayıcı `with` bloğunun içinde olduğu için bağlam yöneticisinin kendi giriş/çıkış maliyeti (`no_grad` veya `inference_mode` nesnesinin açılıp kapanması) ölçüme girmedi; bu maliyet ayrıca ölçülmedi.

Bellek ölçülmedi: venv'de `psutil` kurulu değil ve paket kurmadık. Bellek farkı hakkında bu notta bir iddia yok.

Repo okumasından (bu oturumda C6'da doğrulanacak): `benchmark.py` ve `predict.py` içinde `@torch.inference_mode()` kullanıldığı görülmüştü. Satır numaraları doğrulanmadığı için burada verilmedi.

## C4 - timm ile pretrained MobileNetV3 ve tek görselde tahmin

Deneme scripti: `scratch/c4_pretrained_predict.py`. Yüklenen model `mobilenetv3_small_100.lamb_in1k` (timm 1.0.30), `eval()` modunda ve `inference_mode` içinde çalıştırıldı. `resolve_data_config` çıktısı: `input_size` (3, 224, 224), interpolation bicubic, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), `crop_pct` 0.875; `create_transform` bunu `Resize(256, bicubic, antialias=True)` + `CenterCrop(224)` + normalize olarak kurdu. Örnek görsel `valid\1\image_06739.jpg` (500x628) için çıktı `(1, 1000)` oldu ve tahmin edilen indeks 985; bu indeks ImageNet'e aittir, çiçek sınıfı değildir. Bu ön işlemenin `deliver/predict.py` ile karşılaştırılması C8'e kaldı, burada karşılaştırma yapılmadı.

Kullanım: python scratch/c4_pretrained_predict.py <görsel_yolu>

## C5 - softmax + topk ile top-5

Deneme scripti: `scratch/c5_top5.py` (C4 ile aynı model, ön işleme ve görsel).

**Önemli:** `mobilenetv3_small_100.lamb_in1k` ImageNet-1k'nın 1000 sınıfı için eğitilmiştir, 102 çiçek sınıfı için **eğitilmedi**. Aşağıdaki sonuçlar yalnızca boru hattının (yükleme, ön işleme, `eval()`, `inference_mode`, softmax, topk) çalıştığını gösterir; çiçek veri setindeki doğruluk hakkında hiçbir bilgi vermez. Tek bir görsel kullanıldı.

Softmax toplamı 1.0 çıktı. Top-5 (indeks, olasılık, ad):

| # | İndeks | Olasılık | Ad |
|---|---|---|---|
| 1 | 985 | 0.3983 | daisy |
| 2 | 309 | 0.0378 | bee |
| 3 | 310 | 0.0296 | ant, emmet, pismire |
| 4 | 325 | 0.0192 | sulphur butterfly, sulfur butterfly |
| 5 | 324 | 0.0163 | cabbage butterfly |

İlk beşin olasılık toplamı 0.5011. Adlar, `timm.data.imagenet_info.ImageNetInfo("imagenet-1k")` içindeki `index_to_description` çıktısıdır (C4'teki 985 indeksinin adı bu çıktıdan alındı). Bu çağrının ağ kullanıp kullanmadığı doğrulanmadı. Klasör `1`'in hangi çiçek türü olduğunu bilmiyoruz, bu yüzden "daisy" çıktısı ile veri setindeki sınıf arasında bir bağ kurmuyoruz.

Argüman sürümünün çalıştırmasında HF_TOKEN uyarısı görüldü (ağırlık önbellekte olduğu halde Hub'a erişim denendi); hangi çağrıdan geldiği doğrulanmadı.

Kullanım: python scratch/c5_top5.py <görsel_yolu>



## C6 - Adil gecikme ölçümü: warm-up, synchronize, batch=1

Kaynak: `src/flowers/evaluation/benchmark.py` içindeki `measure_latency_ms` (kod okundu; bu adımda yeni ölçüm yapılmadı).

- **Warm-up:** Ölçümden önce `n_warmup=20` boş çağrı yapılır. İlk çağrılar genelde sonraki çağrılardan yavaş olabilir (ilk kullanım hazırlıkları, önbellekler); bu genel bilgidir, bu projede ayrıca ölçülmedi. Warm-up çağrıları süreye katılmaz.
- **synchronize:** `device` CUDA ise her çağrıdan sonra `torch.cuda.synchronize()` çağrılır, çünkü GPU işleri asenkron çalışır ve senkronize edilmezse ölçülen süre gerçek hesaplama süresini yansıtmaz. CPU'da senkronizasyon yoktur (boş fonksiyon). Bu makinede CUDA yok, bu yol çalıştırılmadı.
- **batch=1:** Girdi `torch.randn(1, 3, H, W)` biçimindedir. Ölçüm tek görsel gecikmesidir (`ms/görüntü`), toplu işlem verimi değildir.
- Ek olarak `torch.inference_mode()`, sabit thread sayısı (`threads=1` varsayılan) ve ortalama/medyan/p95/std raporlanır (`n_runs=200`).

## C7 - Dummy checkpoint ile çıkarım zinciri denemesi

`best.pt` (A kişisinin baseline ağırlığı) projede ve `C:\dev` altında bulunamadı. Bu yüzden zincir rastgele ağırlıklı geçici bir modelle denendi; bu bir doğruluk veya gerçek hız sonucu değildir.

- `scratch/c7_make_dummy_ckpt.py` ile `mobilenetv3_small_100`, 102 sınıf, `pretrained=False` checkpoint üretildi.
- `scripts/export.py` ile TorchScript'e çevrildi: 6.17 MB, `dtype=fp32`.
- `scripts/benchmark.py` (cpu, 1 thread, tag `dummy-c7`) satırı: `mean_ms` 11.299, `p95_ms` 15.741, `size_mb` 6.17, `img_size` 224, `top1` nan.
- Benchmark geçici dizinde çalıştırıldı ve `results/benchmarks.csv` kirletilmedi; geçici dosyalar silindi.
- Tek çalıştırma, tek makine. Gerçek `best.pt` ile tekrarlanmadı.

## C20-C25 - FP16 / INT8 denemeleri ve teknik karar notu

Ortam: CPU, 1 thread, batch=1, torch 2.14.1+cpu. `torch.backends.quantized.supported_engines` yalnızca `['onednn']`. Doğruluk (top1) bu bölümde ölçülmedi; hiçbir varyantın doğruluk etkisi bilinmiyor.

### C20 - FP16 export (TorchScript, `results/benchmarks.csv`)

| Model | dtype | Boyut (MB) | Ortalama (ms) | p95 (ms) |
|---|---|---|---|---|
| mnv3_small | fp32 | 6.20 | 7.194 | 9.895 |
| mnv3_small_fp16 | fp16 | 3.49 | 6.858 | 9.623 |
| mnv3_large | fp32 | 16.52 | 17.169 | 22.116 |
| mnv3_large_fp16 | fp16 | 8.78 | 13.098 | 16.335 |

- Boyut yaklaşık yarıya indi (small 6.20 → 3.49 MB, large 16.52 → 8.78 MB). Tam 2 kat değil, çünkü dosyada ağırlık dışı veriler (graph, `meta.json`) de var.
- **Gecikme beklenen gibi artmadı, azaldı; ama bu fp16 hesabından kaynaklanmıyor olabilir.** `scripts/benchmark.py` CPU'da fp16 modeli `.float()` ile fp32'ye çevirir (satır 17-18), yani ölçülen hesap fp32'dir; yalnızca ağırlıklar fp16'dan fp32'ye geri yuvarlanmış halde yüklenir. Düşüşün nedeni doğrulanmadı: ölçüm gürültüsü (aynı makinede çalıştırmalar arası %10-20 oynama gözlendi, örn. large fp32 TorchScript 17.169 ms, eager 15.010 ms) ya da fp32/fp16 export farkları (fp32 `freeze` edilir, fp16 edilmez) olabilir. Tek çalıştırmalarla "fp16 CPU'da daha hızlı" sonucu çıkarılmamalı.
- fp16'ya yuvarlamanın doğruluk etkisi `scratch/c24_accuracy.py` ile valid kümesinde (818 görsel, CPU) ölçüldü; ağırlıklar fp16'ya çevrilip fp32'ye geri alındı (CPU'daki teslim yolunun hesabı):

| Model | top-1 fp32 | top-1 fp16 (yuvarlanmış) | Tahmin uyumu |
|---|---|---|---|
| mnv3_small | 0.9315 | 0.9303 | 0.9988 |
| mnv3_large | 0.9939 | 0.9939 | 1.0000 |

  small'da fark 0.0012 (818 görselde yaklaşık 1 görsel), large'da fark yok. Yalnızca valid kümesi, tek ölçüm; `test` kümesi ölçülmedi.

### C21 - Dinamik INT8 (`quantize_dynamic`, Linear, qint8)

Deneme scripti: `scratch/c21_c22_quant_experiment.py` ve `scratch/c21_c22_diag.py` (eager model, CSV'ye yazmaz).

| Model | Conv2d param. | Linear param. | Dönüşen Linear | Boyut (state_dict MB) | Ort. ms (fp32 → dinamik) |
|---|---|---|---|---|---|
| mnv3_small | %92.8 | %6.4 | 0 | 6.30 → 6.30 | 9.345 → 9.448 |
| mnv3_large | %96.4 | %3.0 | 0 | 16.71 → 16.71 | 15.010 → 15.970 |

- Modelde tek Linear var: `classifier`. Tipi `timm.layers.linear.Linear` (`nn.Linear`'in alt sınıfı). `quantize_dynamic` hiçbir katmanı dönüştürmedi (dönüşen sayısı 0, boyut değişmedi). Muhtemel neden: dönüşüm eşlemesi tam tip eşleşmesine bakıyor ve timm'in alt sınıfı eşleşmiyor; bu nedeni ayrıca doğrulamadım (kaynak kodu okumadım).
- Dönüşseydi bile etki küçük olurdu: Linear parametrelerin yalnızca %3.0 (large) ve %6.4'ü (small). Kaba tahmin: Linear ağırlıkları 4 bayttan 1 bayta inse large'da ~0.39 MB (~%2.3), small'da ~0.31 MB (~%5) kazanç. Conv2d katmanları dinamik kuantizasyondan etkilenmez (yalnızca Linear/RNN desteklenir).
- Gecikme farkı gürültü düzeyinde (large'da ~1 ms daha yavaş, tek çalıştırma).

### C22 - Statik PTQ INT8 (`static_ptq_int8`, `src/flowers/compression/quantize.py`)

- Fonksiyon FX graph mode kullanır: `calib_loader`'dan `(x, y)` çiftleri bekler, `n_batches` (varsayılan 20) kalibrasyon batch'i çalıştırır, `backend` varsayılanı `"x86"`.
- **Varsayılanla çalışmıyor:** bu ortamda `RuntimeError: quantized engine X86 is not supported`; yalnızca `onednn` destekleniyor. Fonksiyon `backend="onednn"` ile çağrılınca çalıştı (large, çıktı `(1, 102)`).
- **Kalibrasyon verisi rastgele tensördü** (yalnızca çalışabilirlik testi); gerçek kalibrasyon ve doğruluk ölçümü yapılmadı. Gerçek kullanımda eğitim/valid görsellerinden bir `DataLoader` gerekir.
- Rastgele kalibrasyonlu onednn modeli large için ort. 71.513 ms, p95 99.712 ms; aynı eager yolda fp32 15.010 ms idi. Yani bu ortamda ~5 kat yavaş. Nedeni (backend, hardswish/SE gibi katmanların işlenişi, FX grafiği) incelenmedi. Boyutu ölçülmedi.
- `torch.ao.quantization` kullanımdan kalkıyor (DeprecationWarning: 2.10'da kaldırılacak, `torchao` öneriliyor).

### C24-C25 - Kıyaslama ve karar

| Varyant | Boyut (MB) | Gecikme (ms) | Top-1 (valid) | Not |
|---|---|---|---|---|
| FP32 small / large | 6.20 / 16.52 | 7.194 / 17.169 | 0.9315 / 0.9939 | TorchScript, referans |
| FP16 small / large | 3.49 / 8.78 | 6.858 / 13.098 | 0.9303 / 0.9939 | Boyut ~yarı; gecikme farkı fp16 hesabından değil (CPU'da fp32'ye çevrilir) |
| Dinamik INT8 | değişmedi | fark yok | ölçülmedi | Linear dönüşmedi; dönüşse bile Linear payı %3-6 |
| Statik INT8 (onednn, rastgele kalibrasyon) | ölçülmedi | large: 71.5 (fp32 eager 15.0) | ölçülmedi | Yavaşladı; gerçek kalibrasyon yok |

Karar (kural: fayda yoksa uygulama):
- **Dinamik INT8: uygulanmıyor.** Conv ağırlıklı MobileNet'te hedef katman (Linear) parametrelerin %3-6'sı; boyut/hız kazancı ihmal edilebilir ve bu ortamda dönüşüm hiç gerçekleşmedi.
- **Statik INT8: uygulanmıyor (şimdilik).** Varsayılan backend çalışmıyor, onednn ile CPU gecikmesi arttı, doğru kalibrasyon ve doğruluk ölçümü yok. Faydası bu ortamda gösterilemedi.
- **FP16: CPU gecikmesi için fayda yok; yalnızca dosya boyutu için isteğe bağlı.** CPU'da hesap fp32'de yapıldığı için hız kazancı beklenmez. Boyut yarıya iner ve valid doğruluğu pratikte korunur (small -0.0012, large 0), ama kazanç sadece depolama/dağıtım boyutunda.
- **CPU çıkarımı için model: `mobilenetv3_large` FP32 TorchScript (önerilen).** Valid top-1 0.9939, small'a göre +0.0624 puan; bedeli ~2.4 kat gecikme (17.169 vs 7.194 ms, tek görsel, 1 thread) ve ~2.7 kat boyut (16.52 vs 6.20 MB). Bu gecikme tek görsel çıkarımı için küçük sayılabilir ama bir gecikme/boyut hedefi verilmedi; sıkı bir sınır varsa `mnv3_small` (0.9315) alternatiftir. Karar valid kümesine dayanır, test kümesi ve tekrarlı ölçüm yapılmadı.
- Dosya boyutu önemliyse large için FP16 (8.78 MB, aynı doğruluk) eklenebilir; bu bir tercih, CPU hızı için gerekçe değil.

## C26-C30 - Teslimat: `deliver/` ve çıkarım adımları

**Teslim edilen model:** `mobilenetv3_large` FP32 TorchScript (Gün 4 kararı). Proje kökündeki `mobilenetv3_large.pt` dosyası `deliver/flower_model.pt` olarak kopyalandı (README'nin beklediği ad; SHA-256 aynı, 17.325.946 bayt ≈ 16.52 MB). `cat_to_name.json` (102 sınıf, klasör adı → çiçek adı) `deliver/` içine kopyalandı.

`deliver/` içeriği: `predict.py`, `flower_model.pt`, `cat_to_name.json`. Gereken paketler: `torch`, `numpy`, `Pillow` (`timm` ve proje paketi gerekmez).

### predict.py

- Girdi doğrulama (C9/C10 taslağından taşındı): dosya yoksa `FileNotFoundError`; dosya geçerli görsel değilse ya da bozuk/kesikse `InvalidImageError` (`ValueError` alt sınıfı); yol klasörse yine `InvalidImageError`. `UnidentifiedImageError` yanında `OSError` de yakalanır, çünkü kesik dosya `OSError` verir.
- `FlowerClassifier.predict` ve `predict_batch` davranışı aynı (README'deki kullanım bozulmadı). Yeni: `predict_topk(path, k=5)` → `[{"class", "name", "probability"(%)}, ...]`.
- Komut satırı **değişti**: eski `python predict.py model.pt foto.jpg` biçimi yerine `--image` / `--image_dir` (ikisi birbirini dışlar, biri zorunlu). Model yolu `--model` ile verilir, varsayılan `predict.py`'nin yanındaki `flower_model.pt`.
- Windows notu: model dosya nesnesiyle yüklenir (`open(..., "rb")`). `torch.jit.load` yolu `Masaüstü`, `Sınıf` gibi ASCII olmayan karakterler içeren mutlak yollarda açamadı (aynı dosya göreli yolla ve dosya nesnesiyle açıldı).

### Örnek komutlar

```
python deliver/predict.py --image foto.jpg
python deliver/predict.py --image_dir gorseller/ --topk 3
python deliver/predict.py --image foto.jpg --json
python deliver/predict.py --model baska_model.pt --image foto.jpg --device cuda
```

Çıkış kodları: 0 = hepsi başarılı, 1 = en az bir görsel hatalı (klasör modunda hatalı dosyalar akışı durdurmaz, sonda özet yazılır), 2 = model/klasör bulunamadı ya da klasörde desteklenen görsel yok. Klasörde yalnızca `.jpg .jpeg .png .bmp .webp .tif .tiff` taranır.

Python'dan: `from predict import FlowerClassifier; clf = FlowerClassifier("flower_model.pt"); clf.predict_topk("foto.jpg")`.

### Uçtan uca doğrulama (CPU, valid görselleri ve geçici dosyalarla)

| Senaryo | Sonuç |
|---|---|
| `valid/1/image_06739.jpg` | Top-1 sınıf 1 `pink primrose` %60.87 (sonra 52 `wild pansy` %2.18) |
| `valid/10/` içinden bir görsel | Top-1 sınıf 10 `globe thistle` %92.33 |
| Olmayan dosya | `FileNotFoundError` mesajı, çıkış kodu 1 |
| Metin içerikli `.jpg` | `InvalidImageError (UnidentifiedImageError)`, çıkış kodu 1 |
| Kesik görsel | `InvalidImageError (OSError)`, çıkış kodu 1 |
| `--image_dir` (2 geçerli, 2 hatalı, 1 `.txt`) | 2 başarılı, 2 hatalı raporlandı; `.txt` atlandı |
| `--json`, `--image` + `--image_dir` birlikte, argümansız, olmayan klasör | JSON çıktı; son üçü argparse/klasör hatası (çıkış kodu 2) |
| `pytest -q tests` | 4 geçti (değişiklikten önce de 4 geçiyordu) |

Sınırlar: Doğrulama yalnızca CPU'da ve birkaç görselle yapıldı; GPU yolu (`--device cuda`) çalıştırılmadı. İki doğru örnek, genel doğruluk kanıtı değildir (valid top-1 için Gün 4 notuna bakın). Testler `predict.py` için yalnızca ön işlemeyi kapsıyor; yeni CLI ve `predict_topk` için otomatik test eklenmedi.

`.gitignore` içinde `*.pt` olduğu için `deliver/flower_model.pt` git'e otomatik girmez; teslimde ayrıca eklenmesi (ya da başka yolla iletilmesi) gerekir.

## D (kod inceleme) geri bildirimi: normalizasyon, etiket eşlemesi, model seçimi

İnceleme scriptleri: `scratch/d_norm_check.py`, `scratch/d_variants.py` (yalnızca terminale yazar). Model: `mobilenetv3_large` FP32, valid 818 görsel, CPU, `predict.py` ön işlemesi.

### 1. Çifte normalizasyon

- Güncel `src/flowers/export/packager.py` (`_NormWrapper`) normalizasyonu modelin **içine** gömer ve meta'ya `"norm_in_model": True` yazar. Güncel `deliver/predict.py` (`_logits`) bu bayrağa bakar: bayrak varsa dışarıda normalize etmez, yoksa `(x - mean) / std` uygular. Kod yolunda normalizasyon tek yerde yapılıyor, çifte normalizasyon yok.
- **Mevcut `deliver/flower_model.pt` eski export'tan** (güncel packager öncesi): meta'da `norm_in_model` yok, grafta mean/std buffer'ı ve `sub`/`div` izi yok. Yani bu dosyada normalizasyon modelin içinde **değil**; `predict.py` onu dışarıda tek kez yapıyor (doğru çalışıyor, aşağıdaki A).
- **Tutarsızlık:** README "normalizasyon MODELİN İÇİNDE, dışarıda uygulama" diyor, ama mevcut `deliver/flower_model.pt` bunu sağlamıyor. README'ye uyup bu dosyaya ham `[0,1]` girdi veren biri normalizasyonsuz çalıştırır (aşağıdaki D).

| Varyant (valid, 818 görsel) | Top-1 |
|---|---|
| A) Mevcut teslim: eski export + `predict.py` tek normalizasyon | 0.9939 |
| B) Güncel packager ile yeniden export, normalizasyon modelin içinde, dışarıda yok | 0.9939 (A ile logit farkı 0, aynı tahmin %100) |
| C) Güncel export + dışarıda da normalizasyon (**çifte**) | 0.0147 |
| D) Eski export, hiç normalizasyon yok (README sözleşmesi + mevcut dosya) | 0.7555 |

Çifte normalizasyon yakalanması kolay bir hatadır (valid'de ~%1.5'e çöker); mevcut teslim ve güncel kod yolu bu hatayı içermiyor. Önerilen düzeltme: teslim dosyasını güncel packager ile yeniden export etmek (B, A ile birebir aynı sonuç), böylece README ile dosya tutarlı olur. Henüz uygulanmadı.

### 2. Etiket eşlemesi ve shadow düşüşü

- Sınıf klasörleri `ImageFolder` ile **alfabetik** (dize) sıralanır: indeks 0..5 = `1, 10, 100, 101, 102, 11`. `argmax` indeksi klasör numarası değildir ve +1 ofset yanlıştır. Doğru eşleme `meta["idx_to_class"]` (checkpoint'in `class_to_idx` tersi). `predict.py` bunu kullanıyor, ofset yok.
- Kanıt: A varyantı valid'de 0.9939. Naif eşlemeler çöker: `argmax+1 == klasör no` → 0.0880, `argmax == klasör no` → 0.1491. Yani bu hata olsaydı valid'de de görülürdü.
- `results/worst10_shadow_classes.csv`: `class_idx` 9, 10, 13 ↔ klasör 15, 16, 19, alfabetik sıradaki karşılıkla (dizin 9 = "15", 10 = "16", 13 = "19") birebir uyuşuyor. 10 isim `deliver/cat_to_name.json` ile eşleşiyor (0/10 uyuşmazlık); `results/class_to_name_mapping.csv` ile `cat_to_name.json` arasında 0/102 uyuşmazlık. Shadow değerlendirmesinde indeks-klasör eşlemesinde ofset görünmüyor.
- predict.py'nin etiket eşlemesi doğru ve ofset hatası içermemektedir; shadow testteki düşüşün kesin nedeni (harici script normalizasyon hatası mı, yoksa gerçek bir alan kayması / domain shift mi) ilgili script ve görseller depoda bulunmadığı için ayırt edilemedi (yalnızca raporlar ve CSV'ler var; `docs/shadow_test_report.md` kendisi de "doğrulanmadı" uyarısı taşıyor).
- Elde edilen bilgiler: etiket ofseti ve çifte normalizasyon bu repodaki teslim yolunda yok; `predict.py` yolu valid'de 0.9939. Ama shadow'u üreten pipeline'ın hangi normalizasyonu ve eşlemeyi kullandığı bilinmiyor. Hipotezler (doğrulanmadı): shadow scripti `predict.py`/güncel export yerine eski/yanlış bir kombinasyon kullandı (ör. D benzeri normalizasyonsuz çalıştırma valid'de bile 0.7555'e düşürüyor); gerçek dış fotoğraflarda EXIF yönü uygulanmıyor (`predict.py` ve `ImageFolder` `exif_transpose` yapmıyor); gerçek alan kayması. Karar için shadow script ve görseller gerekli: aynı 306 görsel `predict.py` ile çalıştırılıp sonuç karşılaştırılmalı.
- Shadow n = 306 (sınıf başına 3), tek sınıf doğruluğu 0, 1/3, 2/3 ya da 1; küçük örneklem.

### 3. Model seçimi: 0.60·Accuracy + 0.25·Latency_Score + 0.15·Size_Score

**Eşikler henüz kesinleşmedi, bu yüzden aşağıdaki skorlar geçici bir normalizasyonla hesaplandı:** `Latency_Score = en_düşük_gecikme / gecikme`, `Size_Score = en_küçük_boyut / boyut` (en iyi aday = 1.0). Gecikme `results/benchmarks.csv` (CPU, 1 thread, batch=1), boyut TorchScript MB. Accuracy iki kaynakla verildi: valid (ölçüldü) ve shadow (doğrulanmamış, yukarıya bakın).

| Model | Valid acc | Shadow acc | Gecikme (ms) | Boyut (MB) | Latency_Score | Size_Score | Skor (valid acc) | Skor (shadow acc) |
|---|---|---|---|---|---|---|---|---|
| mnv3_small | 0.9315 | 0.1895 | 7.194 | 6.20 | 1.000 | 1.000 | **0.959** | **0.514** |
| mnv3_large | 0.9939 | 0.2680 | 17.169 | 16.52 | 0.419 | 0.375 | 0.757 | 0.322 |
| effnet_b0 | 0.9914 | 0.2451 | 37.100 | 15.76 | 0.194 | 0.393 | 0.702 | 0.255 |
| convnext_nano | 0.9939 | 0.2516 | 98.416 | 57.32 | 0.073 | 0.108 | 0.631 | 0.185 |

**Sonuç:** Bu geçici (en iyiye oranlama) normalizasyonla `mnv3_small` kazanıyor, `mnv3_large` değil. Gün 4'teki "mobilenetv3_large" seçimi bu formül ve bu normalizasyonla gerekçelendirilemiyor. Large'ın kazanması için doğruluk kazancının (+0.0624 valid, ayrıca +0.0785 shadow) hız ve boyut kaybını aşması gerekir. Başabaş koşulu (valid doğruluğu ile): `0.25·(1 − L_large) + 0.15·(1 − S_large) < 0.6·(0.9939 − 0.9315) = 0.0374`. Eşikler small ve large'ı ikisini de tam puanlıyorsa (ör. gecikme ve boyut sınırları ≥ 17.2 ms ve ≥ 16.5 MB ise) large kazanır (0.996 > 0.959). Eşikler kesinleşince bu tablo yeniden hesaplanmalı ve model seçimi ona göre teyit edilmeli. Bu notta nihai seçim değiştirilmedi.

Sınırlar: Gecikmeler tek çalıştırma, geliştirme makinesi (değerlendirme makinesinde farklı çıkar, README de belirtiyor); fp16/int8 varyantları bu tabloda yok; shadow doğrulukları doğrulanmamış.
