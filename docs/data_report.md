# Veri Kümesi İnceleme, Kalite Kontrol ve Sızıntı Raporu

**Tarih:** 26 Ekim 2026  
**Hazırlayan:** B (Veri ve Değerlendirme Sorumlusu)  
**Veri Kümesi:** Udacity Flower Dataset (Oxford 102 Flowers)

---

## 1. Veri Bölümleme ve Genel Sayılar (B1, B2, B3)
Veri kümesindeki 8.189 görsel taranmış ve doğrulanmıştır:

| Split | Sınıf Sayısı | Görsel Sayısı | Durum |
| :--- | :---: | :---: | :--- |
| **Train** | 102 | 6.552 | Doğrulandı |
| **Valid** | 102 | 818 | Doğrulandı |
| **Test** | 102 | 819 | Doğrulandı (Not: Görev tablosundaki 921 yerine veri kümesi orijinal toplamı 819'dur) |
| **Toplam** | **102** | **8.189** | Bütünlük tam |

---

## 2. Sınıf Dağılımı ve Dengesizlik Analizi (B4, B5)
- **Train Dağılımı (B4):** Min: 27, Medyan: 53-54, Maks: 206 görsel. Ciddi bir sınıf dengesizliği (class imbalance) mevcuttur; azınlık sınıfların doğru tahmin oranı (recall) düşebilir.
- **Validasyon Oynaklığı (B5):** Sınıf başına 1 ile 28 görsel düşmektedir (medyan: 7). Bazı sınıflarda yalnızca 1 görsel olduğundan, eğitim esnasında validasyon doğruluğunda (validation accuracy) görülen sert zikzaklar ve dalgalanmalar normaldir.

---

## 3. Veri Kalitesi ve Boyut İncelemesi (B6, B7)
- **Bozuk Dosya Sayısı (verify):** 0 (Tüm dosyalar sağlam).
- **Çok Küçük Görseller (<64 px):** 0 adet.
- **En Küçük Boyut:** 500x500 piksel. Görseller 224x224 için fazlasıyla kalitelidir.

---

## 4. Görsel Çeşitlilik ve Gözlemler (B8)
- Arka planlar hem flu (bokeh) hem de yeşillikli karmaşık zeminler içerir.
- Çiçekler tek renk ve çok renkli alacalı yapılardadır. Model sadece renge değil; taç yaprak formu, doku ve kenar geometrisine odaklanmalıdır.

---

## 5. Veri Büyütme (Augmentation) İncelemesi (B10, B11, B12)
- **RandomResizedCrop:** Çiçeğin farklı yakınlaştırma ve açı varyasyonlarına karşı modeli dayanıklı kılar.
- **RandomHorizontalFlip:** Sağa-sola bakış ezberini kırar ve veri çeşitliliğini artırır.
- **Hue Değişimsiz ColorJitter:** Renk çiçek türü için kritik bir ayırt edici olduğundan Hue (renk tonu) sabit tutulmuş; parlaklık, kontrast ve doygunluk (saturation) ±0,1 aralığında oynatılmıştır; Hue 0'dır.
- **Normalizasyon:** ImageNet standartları olan `mean=[0.485, 0.456, 0.406]` ve `std=[0.229, 0.224, 0.225]` değerleri `timm` ile doğrulanmıştır.

---

## 6. Veri Sızıntısı (Data Leakage) ve MD5 Analizi (B13, B14)
8.189 görselin MD5 parmak izi taranmıştır:
- **Train - Test Çakışması:** 0 adet.
- **Valid - Test Çakışması:** 0 adet.
- **Train - Valid Çakışması:** **1 adet mükerrer çift tespit edildi!**
  - **MD5 Hash:** `d5de43837c54fac145cb6d22fea210d0`
  - **Train Dosyası:** `image_08077.jpg`
  - **Valid Dosyası:** `image_08067.jpg`
- **Aksiyon:** Dosya isimleri farklı olsa da pikseller birebir aynıdır. Validasyon kümesinde yapay başarı şişmesi yaşanmaması için eğitim setindeki `image_08077.jpg` dosyasının çıkarılması önerisiyle Teknik Lider A'ya bildirilmiştir.
- **Çözüm:** Mükerrer görsel koda `exclude_train` ayarıyla eklendi; `image_08077.jpg` yalnızca train'den çıkarılır, valid kopyası kalır. Dosyalar diskten silinmedi.
- **Sınır:** MD5 yalnızca birebir aynı dosyaları yakalar. Aynı çiçeğin farklı açıdan çekilmiş ya da kırpılmış yakın kopyaları ölçülmedi.
