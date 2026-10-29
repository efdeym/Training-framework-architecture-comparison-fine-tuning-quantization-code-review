# Gölge Test (Shadow Test) ve Genelleme Riski Raporu (B28, B31)

- **Tarih:** 29 Ekim 2026
- **Sorumlu:** B - Veri ve Değerlendirme
- **Kapsam:** Bağımsız dış veri kümesi üzerinde 4 aday modelin sıfır-gözlem (out-of-distribution) dayanıklılık testi

---

## 1. Yönetici Özeti ve Temel Skorlar (B26)
Oxford Flowers 102 üzerinde fine-tune edilen modeller, eğitim ve validasyon ortamında %99 seviyesinde başarı göstermiş olsa da, internetten (iNaturalist ve Wikimedia Commons) toplanan 102 sınıf ve 306 bağımsız görsel üzerinde belirgin bir performans düşüşü sergilemiştir:

| Model                | Mimari                |   Valid Top-1 (%) |   Gölge Top-1 (%) |   Genelleme Farkı (%) |
|:---------------------|:----------------------|------------------:|------------------:|----------------------:|
| mnv3_small_full      | mobilenetv3_small_100 |             93.15 |             18.95 |                 74.2  |
| mobilenetv3_full     | mobilenetv3_large_100 |             99.39 |             26.8  |                 72.59 |
| efficientnet_b0_full | efficientnet_b0       |             99.14 |             24.51 |                 74.63 |
| convnext_nano_full   | convnext_nano         |             99.39 |             25.16 |                 74.23 |

---

## 2. Genelleme Riski ve Alan Kayması Yorumu (B28)
Validasyon seti ile gölge test seti arasındaki **~%73'lük başarım farkı** (genelleme uçurumu) şu nedenlere dayanmaktadır:

1. **Fotoğrafik Yanlılık (Photographic Bias):** Oxford Flowers 102 veri setindeki görseller profesyonel veya odaklanmış makro çekimlerden oluşurken; gölge test görselleri vahşi doğada, geniş açılı, yaprak ve çevre örtüsü içeren karmaşık kompozisyonlara sahiptir.
2. **Arka Plan Aşırı Öğrenmesi (Shortcut Learning):** Modeller sadece çiçek morfolojisine değil, Oxford kümesindeki özel yeşil tonlarına ve bulanık (bokeh) arka planlara aşırı uyum sağlamıştır (overfitting).
3. **Model Karşılaştırması:** `mobilenetv3_large_100` (%26.80) hafif mimarisine rağmen karmaşık ortamlarda en yüksek direnci göstermiştir.

---

## 3. En Çok Yanılınan Sınıflar (B30)
Aşağıdaki sınıflarda modeller gerçek dünya fotoğraflarında en düşük performansı kaydetmiştir:

|   folder | flower_name              |   accuracy |
|---------:|:-------------------------|-----------:|
|       15 | yellow iris              |          0 |
|       16 | globe-flower             |          0 |
|       19 | balloon flower           |          0 |
|       21 | fire lily                |          0 |
|       22 | pincushion flower        |          0 |
|       23 | fritillary               |          0 |
|       27 | prince of wales feathers |          0 |
|       29 | artichoke                |          0 |
|       31 | carnation                |          0 |
|       33 | love in the mist         |          0 |

Hatalı tahminlerin görsel ızgarası ve detaylı dağılımı `results/worst10_shadow_predictions.png` adresine kaydedilmiştir.
