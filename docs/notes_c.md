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
