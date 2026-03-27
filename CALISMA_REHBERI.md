# BM 480 - PROJE 2 ÇALIŞMA REHBERİ
## Head CT Hemorrhage Classification
### Terimler, Kararlar, Neden Kullanıldı & Hoca Soruları

---

## 1. PROJENİN AKIŞ ŞEMASI

```
[Veri Seti: 200 CT Görüntü]
         │
         ▼
[1. Ön İşleme (Preprocessing)]
    - Resize (224x224)
    - RGB dönüşümü
    - Normalizasyon (mean/std → SADECE train'den)
         │
         ▼
[2. Veri Bölümleme (Data Split)]
    - Stratified Split: 70% Train / 15% Val / 15% Test
    - Sınıf oranları korunur
    - random_state=42 (tekrarlanabilirlik)
         │
         ▼
[3. Veri Artırımı (Data Augmentation)]
    - SADECE Train setine uygulanır
    - Flip, Rotation, CLAHE, Noise, Elastic
         │
         ▼
[4. Model Mimarisi]
    ├── ConvNeXt-Tiny (Pre-trained, Transfer Learning)
    └── Custom CNN (Özgün Tasarım, 1.2M parametre)
         │
         ▼
[5. Hiperparametre Tuning]
    - Grid Search: LR, Batch Size, Weight Decay
    - Validation seti üzerinde değerlendirilir
         │
         ▼
[6. Eğitim (Training)]
    - AdamW optimizer
    - ReduceLROnPlateau scheduler
    - Early Stopping (patience=7)
    - Checkpoint: En iyi val_loss
         │
         ▼
[7. Değerlendirme (Evaluation)]
    ├── Confusion Matrix
    ├── Accuracy, Precision, Recall, F1
    ├── ROC-AUC Eğrileri
    ├── t-SNE Feature Visualization
    ├── Grad-CAM (Model neye bakıyor?)
    └── Ensemble Karşılaştırma
         │
         ▼
[8. Arayüz (Gradio)]
    - Dosya yükleme → Tahmin + Olasılık + Grad-CAM
```

---

## 2. TERİMLER SÖZLÜĞÜ

### Veri İşleme Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **Stratified Split** | Veriyi bölerken her sette sınıf oranlarını koruyan yöntem | 200 görüntü, 50/50 dağılım → dengenin bozulmaması için zorunlu |
| **Data Leakage** | Test verisinin eğitime sızması | Normalizasyon, augmentation vb. train'den sonraya alınarak önlendi |
| **Augmentation** | Mevcut veriden yeni örnekler türetme | 200 görüntü DL için az → augmentation ile etkili örnek sayısı artırıldı |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | Medikal görüntülerde düşük kontrast sorunu → CLAHE bunu düzeltir |
| **Elastic Transform** | Görüntüye elastik deformasyon uygulama | Medikal görüntülerde doğal anatomik varyasyonları simüle eder |
| **Normalization** | Piksel değerlerini standardize etme (mean=0, std=1) | Farklı CT cihazlarından gelen parlaklık farklarını nötralize eder |

### Model Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **Transfer Learning** | Başka veri setinde öğrenilmiş ağırlıkları kullanma | 200 görüntü sıfırdan eğitim için yetersiz → ImageNet ağırlıkları başlangıç noktası |
| **ConvNeXt** | 2022'de Meta'nın geliştirdiği modern CNN mimarisi | ResNet'in modernize edilmiş hali, ViT ile rekabet edebilir, medikal görüntüde başarılı |
| **Fine-tuning** | Pre-trained modelin son katmanlarını yeni veriyle güncelleme | Tüm ağı eğitmek yerine son katmanları adapte etmek küçük veride daha stabil |
| **BatchNorm** | Her mini-batch'te aktivasyonları normalize etme | Eğitimi hızlandırır, internal covariate shift'i azaltır |
| **Global Average Pooling** | Feature map'leri tek bir vektöre indirger | Fully connected katmana göre çok daha az parametre → overfitting riski düşer |
| **Dropout** | Eğitimde rastgele nöronları devre dışı bırakma | Overfitting önlemi → küçük veri setinde kritik |

### Eğitim Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **AdamW** | Adam optimizer + decoupled weight decay | Adam'ın regularization problemi çözülmüş hali, modern standart |
| **Early Stopping** | Val loss iyileşmezse eğitimi durdurma | Overfitting noktasını otomatik bulur (patience=7) |
| **ReduceLROnPlateau** | Val loss düzleşince LR'yi azaltma | Başta hızlı öğren, sonra ince ayar yap → daha iyi minimum |
| **CrossEntropyLoss** | Sınıflandırma için standart kayıp fonksiyonu | İkili sınıflandırma problemi için uygun |
| **Weight Decay** | Ağırlıklara L2 regularization | Büyük ağırlıkları cezalandır → overfitting önle |

### Değerlendirme Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **Confusion Matrix** | Gerçek vs Tahmin karşılaştırma tablosu | TP, TN, FP, FN dağılımını görsel gösterir |
| **Precision** | Pozitif tahminlerin ne kadarı doğru | "Kanama var" dediğinde ne kadar haklı? |
| **Recall (Sensitivity)** | Gerçek pozitiflerin ne kadarı yakalandı | Gerçek kanamaların ne kadarını buluyor? → MEDİKALDE EN KRİTİK |
| **F1-Score** | Precision ve Recall'ın harmonik ortalaması | İkisinin dengesini tek sayıda özetler |
| **ROC-AUC** | Threshold bağımsız performans ölçüsü | Farklı threshold'larda modelin genel yeteneğini gösterir |
| **Grad-CAM** | Modelin karar verirken baktığı bölgeleri gösteren ısı haritası | Model gerçekten kanama bölgesine mi bakıyor yoksa artifakta mı? |
| **t-SNE** | Yüksek boyutlu feature'ları 2D'de gösterme | Modelin sınıfları ne kadar iyi ayırdığını görsel gösterir |
| **Ensemble** | Birden fazla modelin tahminlerini birleştirme | Farklı modeller farklı hata yapar → birleşince hata azalır |

---

## 3. NEDEN BU KARARLAR ALINDI?

### Neden ConvNeXt (ResNeXt yerine)?
- **ConvNeXt (2022)** ResNet ailesinin en modern versiyonudur
- Vision Transformer (ViT) ile rekabet eden performansa sahip
- Daha iyi gradient akışı ve training stability
- `timm` kütüphanesi ile kolay erişim
- ResNeXt de iyi bir seçim olurdu, ama ConvNeXt daha güncel ve daha iyi sonuç verir

### Neden 70/15/15 split?
- **Ders notu referansı**: Küçük-dengeli veri için 70/15/15 önerilir (Bölüm 2)
- 200 örnek → Test: 30, Val: 30, Train: 140
- Val seti model seçimi için yeterli (her sınıftan ~15 örnek)
- Test seti final rapor için yeterli

### Neden Augmentation sadece Train'de?
- **Altın kural** (Ders notu Bölüm 6): "Split first, then preprocess"
- Val/Test setleri gerçek dünya dağılımını temsil etmeli
- Augmented veri val/test'e sızarsa performans yapay olarak şişer
- Aynı görüntünün augmented versiyonları farklı setlere düşerse → **data leakage**

### Neden Normalizasyon sadece Train'den hesaplanır?
- **Data leakage önleme** (Ders notu Bölüm 5): "Scaler, imputer, encoder sadece train'den fit edilmeli"
- Test verisinin istatistikleri eğitim sürecine sızmamalı
- Gerçek dünyada deployment'ta test verisi önceden bilinmez

### Neden Custom CNN ~1.2M parametre?
- 200 görüntü için 28M parametreli ConvNeXt bile çok büyük
- Custom CNN daha az parametre = daha az overfitting riski
- Transfer learning ile ConvNeXt avantajlı başlar ama Custom CNN'in generalizasyonu farklı olabilir
- İkisini karşılaştırmak projenin amacı

### Neden Ensemble?
- **Medikal AI'da güvenilirlik kritik**: Yanlış negatif = kaçırılan kanama = hayat tehlikesi
- İki farklı mimari farklı hata paterni → birleşince hatalar azalır
- Soft voting: Olasılıkların ağırlıklı ortalaması → daha kalibre sonuçlar
- Optimal ağırlıklar validation seti üzerinden hesaplanır

### Neden Grad-CAM?
- **Explainability (Açıklanabilirlik)**: Medikal AI'da "kara kutu" kabul edilemez
- Doktor modelin kararını anlamak ister
- Model gerçekten kanama bölgesine mi bakıyor? Yoksa CT'nin kenarındaki artifakta mı?
- Yanlış tahminlerde bile model hangi bölgeye odaklanmış görmek öğretici

---

## 4. HOCA SORULARI ve CEVAPLAR

### Veri İşleme Soruları

**S: Neden random split yerine stratified split kullandınız?**
C: Veri setimiz küçük (200 örnek). Random split'te bir sınıf test setinde az temsil edilebilir. Stratified split her sette sınıf oranlarını korur (%50 Normal / %50 Hemorrhage). Bu, özellikle küçük veri setlerinde güvenilir performans tahmini için zorunludur.

**S: Data augmentation'ı split'ten önce yapsaydınız ne olurdu?**
C: Aynı orijinal görüntünün augmented versiyonları hem train hem test setine düşebilirdi. Bu "data leakage" oluşturur. Model aslında test verisini "görmüş" olur ve performans yapay olarak yüksek çıkar. Gerçek dünyada bu performans elde edilemez.

**S: Neden CLAHE kullandınız?**
C: CT görüntüleri genellikle düşük kontrastlıdır. CLAHE (Contrast Limited Adaptive Histogram Equalization) lokal kontrastı artırır, bu sayede kanama bölgeleri daha belirgin hale gelir. "Adaptive" olması farklı bölgelere farklı kontrast uygulamasını sağlar.

**S: Mean ve std'yi neden tüm veri setinden değil de sadece train'den hesapladınız?**
C: Bu data leakage önlemenin temel kuralıdır. Gerçek dünyada deployment'ta test verisini önceden göremeyiz. Eğer test verisinin istatistiklerini kullanırsak, modelin gerçek performansını doğru ölçemeyiz. Train seti "bilinen dünya"dır, normalizasyon parametreleri oradan gelmelidir.

### Model Soruları

**S: ConvNeXt'i neden seçtiniz? ResNet50 kullansaydınız?**
C: ConvNeXt (2022) Meta AI tarafından geliştirilmiş, ResNet ailesinin en modern versiyonudur. Vision Transformer'lar ile rekabet eden performansa sahiptir. ResNet50 de kullanılabilirdi, ancak ConvNeXt daha iyi gradient akışı, daha modern mimari blokları (depthwise conv, GELU, Layer Norm) ve daha iyi training dynamics sunar.

**S: Custom CNN'iniz neden 4 blok?**
C: 224x224 giriş boyutu, her blokta 2x downsample: 224→112→56→28→14. 4 bloktan sonra 14x14 feature map kalır, bu yeterli uzamsal bilgi içerir. 5. blok eklemek 7x7'ye düşürür ve küçük veri setinde overfitting riskini artırır. Kanal sayısı (32→64→128→256) progressif artışla derinleştikçe daha soyut özellikler öğrenir.

**S: Neden Global Average Pooling kullandınız?**
C: Flatten + Dense yerine GAP kullandık çünkü: (1) Parametresizdir - overfitting riski azalır, (2) Spatial bilgiyi doğal şekilde özetler, (3) Input boyutu değişse bile çalışır, (4) Grad-CAM gibi görselleştirme tekniklerinde daha iyi sonuç verir.

**S: Transfer learning ile sıfırdan eğitim arasındaki fark?**
C: Transfer learning ImageNet'te öğrenilmiş kenar, doku, şekil gibi genel özellikleri başlangıç noktası olarak kullanır. 200 görüntüyle sıfırdan bu özellikleri öğrenmek neredeyse imkansızdır. ConvNeXt zaten bu özellikleri bilir, biz sadece "CT'de kanama var/yok" farkını öğretiyoruz.

### Eğitim Soruları

**S: Early stopping'in patience'ı neden 7?**
C: Çok küçük patience (2-3) → eğitim erken durabilir, model tam öğrenemez. Çok büyük patience (15+) → overfitting başladıktan sonra bile eğitime devam eder. 7, küçük veri setleri için yaygın bir değerdir. LR scheduling ile birlikte kullanıldığında, LR düştükten sonra birkaç epoch daha deneme şansı verir.

**S: AdamW ile Adam arasındaki fark nedir?**
C: Klasik Adam'da weight decay L2 regularization ile karıştırılır - bu teknik olarak yanlıştır. AdamW weight decay'i optimizer'dan ayırır (decoupled weight decay), böylece regularization etkisi daha doğru uygulanır. 2019'dan beri modern DL projelerinde standart optimizer'dır.

**S: Modeliniz overfit ettiğini nasıl anlarsınız?**
C: Training analysis grafiğinde: (1) Train loss düşerken val loss yükselmeye başlarsa, (2) Train accuracy %100'e yaklaşırken val accuracy düşük kalırsa, (3) Generalization gap (train_acc - val_acc) > %10 ise model ezberliyor demektir. Bu projede early stopping ve dropout ile önlem aldık.

### Değerlendirme Soruları

**S: Neden sadece accuracy değil de F1, Recall da raporladınız?**
C: Medikal uygulamalarda accuracy yanıltıcı olabilir. Örneğin %90 normal / %10 hasta veri setinde "her şeye normal de" diyen model %90 accuracy alır ama hiç hasta bulamaz. Recall (sensitivity) gerçek hastaların ne kadarını yakaladığımızı gösterir - medikal AI'da en kritik metriktir.

**S: ROC-AUC neden önemli?**
C: ROC-AUC threshold'dan bağımsızdır. Farklı threshold değerlerinde modelin genel ayrım yeteneğini ölçer. AUC=0.5 rastgele, AUC=1.0 mükemmel. İki modeli karşılaştırırken daha güvenilir bir metrik sunar çünkü tek bir threshold'a bağlı değildir.

**S: Grad-CAM size ne söylüyor?**
C: Grad-CAM modelin "neye bakarak karar verdiğini" gösterir. İyi bir model kanama olan bölgede yüksek aktivasyon göstermeli. Eğer model CT'nin kenarına, etikete veya artifakta bakıyorsa, shortcut learning yapmış demektir - bu model gerçek dünyada çalışmaz. Grad-CAM bu tür sorunları tespit etmemizi sağlar.

**S: Ensemble neden tek modelden daha iyi?**
C: İki farklı mimari (ConvNeXt: derin ve genel; Custom CNN: sığ ve yerel) farklı hata patternleri gösterir. Bir modelin yanlış tahmin ettiği örneği diğeri doğru tahmin edebilir. Soft voting ile olasılıkların ortalaması alındığında, hatalar dengelenir ve genel doğruluk artar. Bu özellikle medikal uygulamalarda "ikinci görüş" prensibiyle örtüşür.

### Genel Sorular

**S: Bu proje gerçek hayatta kullanılabilir mi?**
C: Hayır, birkaç nedenle: (1) 200 görüntü gerçek tıbbi AI için çok az, binlerce görüntü gerekir, (2) Tek bir veri kaynağı - farklı hastaneler, farklı CT cihazları ile genelleme test edilmeli, (3) Klinik onay (FDA/CE) süreci geçilmeli, (4) Radyolog ile birlikte kullanılmalı, tek başına teşhis aracı olamaz.

**S: Veri seti daha büyük olsaydı neyi değiştirirdiniz?**
C: (1) Split oranını 80/10/10 veya 90/5/5 yapardım - büyük veride %5 bile yeterli test örneği verir, (2) K-fold CV yerine tek split yeterli olurdu, (3) Daha agresif augmentation + mixup/cutmix denerdim, (4) Custom CNN'i daha derin yapabilirdim (overfitting riski azalır), (5) External test seti (farklı hastane verisi) kullanırdım.

**S: GPU yoksa ne olur?**
C: Projemiz CPU'da da çalışır (config.py otomatik algılar). Ancak eğitim çok yavaş olur. ConvNeXt-Tiny 28M parametre, 200 görüntüde bile CPU'da epoch başına ~30-60 saniye sürebilir. GPU ile bu 2-5 saniyeye düşer.

---

## 5. DOSYA YAPISI ve AÇIKLAMALAR

```
DÖ-2/
├── main.py                     ← Ana pipeline, tek komutla her şeyi çalıştır
├── requirements.txt            ← pip install -r requirements.txt
├── CALISMA_REHBERI.md          ← BU DOSYA (sınav çalışma notu)
│
├── src/                        ← Tüm kaynak kodlar
│   ├── config.py               ← Yollar, sabitler, hiperparametre defaults
│   ├── data_preprocessing.py   ← Veri yükleme, normalize, Dataset sınıfı
│   ├── data_split.py           ← Stratified train/val/test bölümleme
│   ├── data_augmentation.py    ← Augmentation pipeline (albumentations)
│   ├── custom_cnn.py           ← Özgün CNN mimarisi (ConvBlock × 4)
│   ├── pretrained_model.py     ← ConvNeXt-Tiny (timm ile yükleme)
│   ├── train.py                ← Eğitim döngüsü, early stopping, grafik
│   ├── evaluate.py             ← Test metrikleri + tüm görselleştirmeler
│   ├── hyperparameter_tuning.py← Grid Search (LR, batch, weight decay)
│   ├── ensemble.py             ← Soft Voting Ensemble + optimal ağırlık
│   ├── gradcam.py              ← Grad-CAM ısı haritası üretimi
│   ├── visualizations.py       ← t-SNE, ROC-AUC, eğitim analizi
│   ├── app.py                  ← Gradio arayüzü (tahmin + Grad-CAM)
│   └── web_crawler.py          ← Harici görüntüleri test etme
│
├── head_ct/head_ct/            ← 200 CT görüntüsü (000.png - 199.png)
├── labels.csv                  ← Etiketler (id, hemorrhage: 0/1)
├── models/                     ← Eğitilmiş modeller (.pth)
├── results/                    ← Tüm grafikler ve metrikler
└── web_crawled_test/           ← Sunum için harici test görüntüleri
```

---

## 6. ÇALIŞTIRMA KILAVUZU

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Tüm pipeline'ı çalıştır (augmentation + tuning + train + eval)
python main.py

# 3. Sadece eğitim
python main.py --train

# 4. Sadece değerlendirme (eğitim sonrası)
python main.py --eval

# 5. Arayüzü başlat
python main.py --app
# Tarayıcıda http://localhost:7860 adresine git

# 6. Sunum için web-crawled görüntüleri test et
# Önce web_crawled_test/ klasörüne CT görüntüleri koy
python main.py --webcrawl
```

---

## 7. RAPOR İÇİN ÜRETİLEN GRAFİKLER

| Grafik | Dosya | Ne İçin |
|--------|-------|---------|
| Augmentation örnekleri | `augmentation_preview.png` | Veri artırımı bölümü |
| Veri seti dağılımı | `dataset_overview.png` | Split stratejisi bölümü |
| ConvNeXt eğitim eğrileri | `convnext_training_curves.png` | Training bölümü |
| Custom CNN eğitim eğrileri | `custom_cnn_training_curves.png` | Training bölümü |
| ConvNeXt eğitim analizi | `convnext_tiny_training_analysis.png` | Overfitting analizi |
| Custom CNN eğitim analizi | `custom_cnn_training_analysis.png` | Overfitting analizi |
| ConvNeXt confusion matrix | `convnext_tiny_confusion_matrix.png` | Sonuçlar bölümü |
| Custom CNN confusion matrix | `custom_cnn_confusion_matrix.png` | Sonuçlar bölümü |
| Ensemble confusion matrix | `ensemble_confusion_matrix.png` | Sonuçlar bölümü |
| Model karşılaştırma | `model_comparison.png` | Sonuçlar bölümü |
| ROC-AUC eğrileri | `roc_auc_curves.png` | Sonuçlar bölümü |
| ConvNeXt t-SNE | `convnext_tiny_tsne.png` | Analiz bölümü |
| Custom CNN t-SNE | `custom_cnn_tsne.png` | Analiz bölümü |
| ConvNeXt Grad-CAM | `convnext_tiny_gradcam.png` | Yorumlanabilirlik |
| Custom CNN Grad-CAM | `custom_cnn_gradcam.png` | Yorumlanabilirlik |
| Grid Search ConvNeXt | `convnext_grid_search.png` | HPO bölümü |
| Grid Search Custom CNN | `custom_cnn_grid_search.png` | HPO bölümü |

---

## 8. PROJEYİ DİĞERLERİNDEN AYIRAN ÖZELLİKLER

1. **Grad-CAM**: Çoğu öğrenci bunu eklemiyor. Model nereye bakıyor görsel olarak göstermek medikal AI'da zorunlu.

2. **Ensemble**: Tek model yerine iki modeli birleştirip karşılaştırma yapmak daha profesyonel.

3. **t-SNE**: Feature space'i görselleştirmek modelin ne öğrendiğini anlatmanın en iyi yolu.

4. **ROC-AUC + PR Curves**: Sadece accuracy raporlamak amatörce. Profesyonel projeler threshold-bağımsız metrikler kullanır.

5. **Eğitim Dinamikleri Analizi**: Overfitting gap grafiği ve LR schedule takibi.

6. **Data Leakage Farkındalığı**: Normalizasyon, augmentation, split sıralaması bilinçli ve kurala uygun.

7. **Arayüzde Grad-CAM**: Sadece tahmin değil, "neden bu tahmin" sorusuna da cevap veren arayüz.
