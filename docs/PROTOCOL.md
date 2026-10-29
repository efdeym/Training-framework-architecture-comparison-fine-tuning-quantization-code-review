# Karşılaştırma Protokolü

Bu doküman, adayların nasıl eğitildiğini, ölçüldüğünü ve seçildiğini sabitler. Tüm sayılar bu kurallara göre üretilir.

## 1. Skor ve sınırlar (kılavuzdan)
Skor = 0,60 × Doğruluk + 0,25 × min(100, 50/T × 100) + 0,15 × min(100, 35/S × 100)
T = CPU ms/görsel (Intel i7, 16 GB, GPU yok), S = kaydedilen model dosyası (MB).
T ≤ 50 ms ve S ≤ 35 MB olunca hız ve boyut puanı tavan yapar. Bu sınırlar içinde karar değişkeni doğruluktur.
Tmax ve Smax önerilmiş değerlerdir, hoca onaylayacak. Zamanlama yöntemi (ön işleme, batch, warm-up) hocanın yönergesine göre güncellenir.

## 2. Eğitim
- Seed 42, giriş 224, tüm adaylar aynı tarif: 30 epoch, ilk 3 epoch yalnızca kafa (LP-FT).
- Model-spesifik farklar yalnızca `configs/<model>.yaml` içinde, burada belgelenir (ConvNeXt-nano: MixUp/CutMix).
- Her koşu yeni `exp_name` ile başlar, `--force` kullanılmaz.

## 3. Doğruluk ölçümü
- Tek referans: `scripts/evaluate.py`. Logdaki test sayısı kullanılmaz (son epoch ağırlıklarıdır).
- Ağırlık: `best.pt`. Bozuk/ezilmiş olduğu bilinen koşuda `last.pt`, tabloda belirtilir (mnv3_small_ft).
- Valid 818 görsel, 1 görsel = %0,12. 4 görselden (~%0,5) küçük farklar ayırt edici sayılmaz.
- Genelleme için gölge test (`data/shadow_test`) esas alınır.

## 4. Boyut ve gecikme ölçümü
- Format: TorchScript, fp32 (`scripts/export.py`).
- Cihaz: CPU, batch=1, `n_warmup=20`, `n_runs>=100` (varsayılan 200), aynı makine.
- Thread: 1 thread ve varsayılan thread sayısı ayrı ayrı ölçülür, kötü olan raporlanır.
- Ölçüm sırasında eğitim veya ağır iş çalışmaz.
- Raporlanan: ortalama, medyan, p95 (ms). Ayrıca ön işleme dahil uçtan uca süre.
- Boyut: dosya bayt cinsinden kaydedilir, MB = bayt / 1e6 (daha büyük sayı) raporlanır.
- Dummy/rastgele ağırlıkla üretilen hiçbir sayı sonuç tablosuna girmez.

## 5. Seçim kuralı
1. Zorunlu: S ≤ 35 MB ve T ≤ 50 ms (uçtan uca, hem 1 thread hem varsayılan). Güvenlik payı için hedef S ≤ 30 MB, T ≤ 35 ms.
2. Şartı sağlayanlar arasında gölge test top-1 en yüksek olan seçilir.
3. Fark ayırt edici değilse (yaklaşık 4 görsel) daha küçük ve daha hızlı olan seçilir.
4. TTA veya ensemble yalnızca gölge testte ölçülen kazanç ve uçtan uca süre/boyut sınırı içinde kalırsa kullanılır.