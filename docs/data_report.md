# Veri Kümesi İnceleme ve Kalite Kontrol Raporu

**Tarih:** 25 Ekim 2026  
**Hazırlayan:** B-Karya DEMİRCİLER (Veri ve Değerlendirme Sorumlusu)  
**Veri Kümesi:** Udacity Flower Dataset (Oxford 102 Flowers türevi)

---

## 1. Veri Bölümleme ve Genel Sayılar (B1, B2, B3)
Veri kümesinde toplam **8.189** adet görsel taranmış ve doğrulanmıştır:

| Split | Sınıf Sayısı | Görsel Sayısı | Durum |
| :--- | :---: | :---: | :--- |
| **Train** | 102 | 6.552 | Doğrulandı |
| **Valid** | 102 | 818 | Doğrulandı |
| **Test** | 102 | 819 | Doğrulandı (Not: Tablodaki 921 hedefi yerine orijinal veri kümesi toplamı olan 819 mevcuttur) |
| **Toplam** | **102** | **8.189** | Bütünlük tam |

---

## 2. Sınıf Dağılımı ve Dengesizlik Analizi (B4, B5)

### Train Kümesi (B4):
- **Minimum Görsel:** 27
- **Medyan Görsel:** 53 - 54
- **Maksimum Görsel:** 206
- **Gözlem:** Ciddi bir sınıf dengesizliği (class imbalance) mevcuttur. Bazı sınıflar 200'ün üzerinde örneğe sahipken bazıları yalnızca 27 örnek barındırmaktadır. Bu durum azınlık sınıflarda recall düşüklüğüne yol açabilir.

### Valid Kümesi ve Oynaklık Uyarısı (B5):
- **Aralık:** Sınıf başına 1 ile 28 görsel.
- **Medyan:** 7 görsel.
- **Teknik Değerlendirme:** Validasyon kümesinde bazı sınıflarda yalnızca **1 adet görsel** bulunmaktadır. Bu durum eğitim sırasında validasyon doğruluğunda (validation accuracy) dalgalanmalara ve zikzaklara yol açacaktır; zira tek bir yanlış tahmin o sınıfın başarımını anında %0'a düşürmektedir. Doğrulama eğrisindeki ufak düşüşler modelin çöktüğü anlamına gelmez.

---

## 3. Veri Kalitesi ve Boyut İncelemesi (B6, B7)
- **Bozuk Dosya Sayısı (verify):** 0 (Tüm dosyalar sağlam).
- **Çok Küçük Görseller (<64 px):** 0 adet.
- **En Küçük Boyut:** 500x500 piksel.
- **Değerlendirme:** Veriler yüksek çözünürlüklü ve temizdir. 224x224 piksel ölçeklemesinde herhangi bir piksel bozulması veya bilgi kaybı riski bulunmamaktadır.

---

## 4. Görsel Çeşitlilik ve Niteliksel Gözlemler (B8)
- **Arka Plan:** Hem bokeh (flu/bulanık odaklı) hem de doğal yeşillikli karmaşık zeminler bulunmaktadır.
- **Renk ve Morfoloji:** Tek renkli çiçeklerin yanı sıra alacalı/çok renkli çiçekler mevcuttur. Model yalnızca renge odaklanmamalı; taç yaprak formu, doku ve kenar geometrisini öğrenecek şekilde eğitilmelidir.
