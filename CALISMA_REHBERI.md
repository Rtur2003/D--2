# BM 480 - PROJE 2 ÇALIŞMA REHBERİ
## Head CT Hemorrhage Classification
### Terimler, Kararlar, Neden Kullanıldı & Hoca Soruları

> **Not:** Bu rehber hem savunma sırasında soruları yanıtlamak için "cep kitabı" niteliğinde hem de kod tabanını anlamayan birine projeyi baştan sona anlatabilecek detayda yazılmıştır. Her kararın *ne* olduğu kadar *neden* olduğu da açıklanır; her açıklamanın sonunda olası soruya direkt cevap vardır.

---

## 0. PROJE ŞARTNAMESİ UYUM KONTROLÜ (Proje2.docx)

Aşağıda hocanın dokümanda istediği her madde ve projede nasıl karşılandığı madde madde eşleştirilmiştir. Savunmada doğrudan bu tabloyu gösterebilirsiniz.

| # | Şartnamede İstenen | Projede Karşılığı | Dosya / Kanıt |
|---|-------------------|-------------------|---------------|
| 1 | Kaggle felipekitamura head-ct-hemorrhage veri seti | 200 görüntü (100 Normal / 100 Hemorrhage) kullanıldı | `head_ct/head_ct/`, `labels.csv` |
| 2 | Sınıflandırma problemi | Binary classification (Normal vs Hemorrhage) | `config.py::CLASS_NAMES` |
| 3 | Veri seti hazırlama (ön işleme / preprocessing) | Resize 224x224, RGB, train'den hesaplanan mean/std ile normalizasyon | `src/data_preprocessing.py` |
| 4 | Veri artırımı (data augmentation) | Albumentations: HorizontalFlip, VerticalFlip, Affine (translate/scale/rotate), RandomBrightnessContrast, CLAHE, GaussianBlur/MedianBlur, GaussNoise, ElasticTransform — **yalnızca train'e** | `src/data_augmentation.py` |
| 5 | Train + Validation + Test split | Stratified 70/15/15 → 139 / 31 / 30; random_state=42 | `src/data_split.py` |
| 5a | Sunumda kullanmak için web-crawling ile birkaç harici görüntü | `web_crawler.py` ve `--webcrawl` modu; `web_crawled_test/` klasörü | `src/web_crawler.py`, `main.py --webcrawl` |
| 6 | 1 pretrained CNN (ConvNeXt, ResNeXt,...) | ConvNeXt-Tiny (timm), ImageNet ön-eğitimli, Progressive Unfreezing ile fine-tune | `src/pretrained_model.py`, `src/train.py` |
| 7 | 1 özgün CNN (kendi tasarımınız) | Custom CNN v2: Stem + Multi-Scale Block + 3x Residual-SE Block + GAP + FC (≈1.3M parametre) | `src/custom_cnn.py` |
| 8 | Hyperparameter tuning (LR, batch size, epoch, early stopping patience, vs.) | Grid Search: LR × batch_size × weight_decay (12 kombinasyon). Sonuçlar tablo ve görsel olarak raporda. | `src/hyperparameter_tuning.py`, `results/*_grid_search.png`, `results/*_best_hparams.json` |
| 9 | Her iki CNN için train/val eğitim grafikleri | Her model için loss & accuracy eğrileri + overfitting analizi | `results/convnext_training_curves.png`, `results/custom_cnn_training_curves.png`, `results/*_training_analysis.png` |
| 10 | Eğitilmiş modellerin saklanması | `.pth` dosyası olarak checkpoint (model_state_dict + metadata) | `models/convnext_tiny_best.pth`, `models/custom_cnn_best.pth` |
| 11 | Overfitting trend kontrolü | Training analysis grafiği: gap eğrisi + renkli uyarı (>%10 kırmızı, >%5 turuncu) | `src/visualizations.py::plot_training_analysis` |
| 12 | Confusion Matrix, Accuracy, Precision, Recall | 3 model için de (ConvNeXt, Custom CNN, Ensemble) ayrı ayrı rapor | `src/evaluate.py`, `results/*_confusion_matrix.png`, `results/*_metrics.json` |
| 13 | Dosya seçimi ile tahmin eden arayüz + olasılık skoru | Gradio arayüzü: görüntü yükle → 3 model seçimi → olasılık + Grad-CAM + detay raporu | `src/app.py` |
| 14 | IEEE Xplore makale formatında rapor | Proje raporu `docs/` klasöründe template'e göre yazılmalı (kod kapsamı dışında) | `project_template.docx` |
| 15 | Akış şeması | Bu rehberin 1. bölümünde + rapora eklenmek üzere hazırlandı | `CALISMA_REHBERI.md::1` |
| 16 | 5 adet IEEE Xplore 2025 makalesi (literatür özeti tablosu) | Bölüm 11'de arama stratejisi ve iskelet sunuldu, 5 makale seçilecek | `CALISMA_REHBERI.md::11` |
| 17 | Referanslar | Rapor sonunda IEEE stiliyle; literatür tablosundaki 5 makale + ConvNeXt + Custom CNN teknikleri | Rapor |
| 18 | Kodlar çalışır durumda + requirements.txt | `main.py` tek komutla pipeline; `requirements.txt` tüm bağımlılıkları içerir | `main.py`, `requirements.txt` |

**Öne çıkan EKSTRA (şartname istemedi ama profesyonellik için eklendi):**
- **Ensemble** (Soft Voting, validation'dan optimal ağırlık)
- **Grad-CAM** açıklanabilirlik (hem evaluate.py'de hem arayüzde)
- **t-SNE** feature space görselleştirmesi
- **ROC-AUC + Precision-Recall** eğrileri
- **Mixup + Label Smoothing + Cosine Annealing + Gradient Clipping + Progressive Unfreezing** (modern eğitim teknikleri)

---

## 1. PROJENİN AKIŞ ŞEMASI

```
[Veri Seti: 200 CT Görüntü (100 Normal, 100 Hemorrhage)]
         │
         ▼
[1. Ön İşleme (Preprocessing)]
    - Resize (224x224)
    - RGB dönüşümü
    - Normalizasyon (mean/std → SADECE train'den hesaplanır)
         │
         ▼
[2. Veri Bölümleme (Data Split)]
    - Stratified Split: 70% Train (139) / 15% Val (31) / 15% Test (30)
    - Sınıf oranları korunur (~%50/%50)
    - random_state=42 (tekrarlanabilirlik)
         │
         ▼
[3. Veri Artırımı (Data Augmentation)]
    - SADECE Train setine uygulanır
    - Flip, Rotation, CLAHE, Noise, Elastic, Affine
         │
         ▼
[4. Model Mimarisi]
    ├── ConvNeXt-Tiny (Pre-trained, 28M parametre, Transfer Learning)
    └── Custom CNN v2 (Özgün: Residual + SE Attention + MultiScale, ~1.3M parametre)
         │
         ▼
[5. Hiperparametre Tuning]
    - Grid Search: LR × Batch Size × Weight Decay (12 kombinasyon)
    - Validation seti üzerinde değerlendirilir (test ASLA kullanılmaz)
         │
         ▼
[6. Eğitim (Training) - İleri Teknikler]
    - AdamW optimizer + Gradient Clipping (max_norm=1.0)
    - Label Smoothing (0.05-0.1) → overconfident tahminleri önler
    - Mixup (alpha=0.2) → veri artırımı + regularization
    - Cosine Annealing (warm restart) → daha yumuşak LR azaltma
    - Progressive Unfreezing (ConvNeXt: önce head, sonra backbone)
    - Early Stopping (patience=5-8)
    - Checkpoint: En iyi val_loss modeli kaydedilir
         │
         ▼
[7. Değerlendirme (Evaluation)]
    ├── Confusion Matrix (3 model)
    ├── Accuracy, Precision, Recall, F1 (weighted + per-class)
    ├── ROC-AUC + Precision-Recall Eğrileri
    ├── t-SNE Feature Visualization (sınıf ayrımı kalitesi)
    ├── Grad-CAM (Model neye bakıyor? Açıklanabilirlik)
    ├── Eğitim Dinamikleri Analizi (overfitting gap, LR schedule)
    └── Ensemble Karşılaştırma (optimal ağırlık: validation'dan)
         │
         ▼
[8. Arayüz (Gradio)]
    - Dosya yükleme → Tahmin + Olasılık + Grad-CAM
    - 3 model seçeneği + detaylı analiz raporu
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
| **Progressive Unfreezing** | Önce sadece classifier eğit, sonra tüm ağı aç | 200 görüntüyle 28M parametreyi direkt eğitmek = overfitting. Kademeli adaptasyon bunu önler |
| **Fine-tuning** | Pre-trained modelin katmanlarını yeni veriyle güncelleme | Tüm ağı eğitmek yerine kademeli adapte etmek küçük veride daha stabil |
| **Residual Connection** | Giriş + çıkış bağlantısı (skip connection) | Gradient vanishing problemi çözer, derin ağları eğitilebilir yapar. "En kötü ihtimal identity öğren" |
| **SE Block (Squeeze-and-Excitation)** | Kanal bazlı attention mekanizması | Her feature kanalının önemini öğrenir → kanama tespiti için hangi kanallar kritik, model bunu seçer |
| **Multi-Scale Feature Fusion** | Farklı çözünürlüklerde (1x1, 3x3, 5x5) özellik çıkarma | CT'de kanama hem küçük (subdural) hem büyük (intracerebral) olabilir → farklı ölçeklerde bakmalıyız |
| **BatchNorm / LayerNorm** | Aktivasyonları normalize etme | Eğitimi hızlandırır, internal covariate shift'i azaltır |
| **Global Average Pooling** | Feature map'leri tek bir vektöre indirger | Fully connected katmana göre çok daha az parametre → overfitting riski düşer |
| **Dropout / Dropout2d** | Eğitimde rastgele nöronları/kanalları devre dışı bırakma | Overfitting önlemi → küçük veri setinde kritik. Conv bloklarda %10-20, classifier'da %40 |

### Eğitim Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **AdamW** | Adam optimizer + decoupled weight decay | Adam'ın regularization problemi çözülmüş hali, modern standart |
| **Label Smoothing** | Hedef etiketi [1,0] yerine [0.95, 0.05] yapma | Modelin %100 emin olmasını engeller → overconfident tahminleri önler, generalizasyonu artırır |
| **Mixup** | İki görüntüyü λ oranında karıştırarak yeni örnek üretme | x_mix = λ*x_i + (1-λ)*x_j. Karar sınırlarını yumuşatır, overfitting azaltır. Zhang et al. (2018) |
| **Cosine Annealing** | LR'yi kosinüs fonksiyonuyla azaltma + warm restart | ReduceLROnPlateau'dan daha yumuşak, warm restart ile yerel minimumlardan kaçabilir |
| **Gradient Clipping** | Gradient normunu max_norm ile sınırlama | Gradient patlamasını önler → eğitim stabilitesi, özellikle küçük batch'lerde önemli |
| **Early Stopping** | Val loss iyileşmezse eğitimi durdurma | Overfitting noktasını otomatik bulur (ConvNeXt: patience=5, Custom CNN: patience=8) |
| **CrossEntropyLoss** | Sınıflandırma için standart kayıp fonksiyonu | İkili sınıflandırma problemi için uygun, label smoothing ile birlikte kullanıldı |
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

### Neden Custom CNN v2 (~1.3M parametre)?
- 200 görüntü için 28M parametreli ConvNeXt bile çok büyük
- Custom CNN daha az parametre = daha az overfitting riski
- **v2 Yenilikleri**: Residual bağlantılar (gradient flow), SE Attention (kanal bazlı önem), Multi-Scale (farklı boyut kanama tespiti)
- Transfer learning ile ConvNeXt avantajlı başlar ama Custom CNN tamamen sıfırdan eğitiliyor
- İkisini karşılaştırmak projenin amacı: pre-trained vs scratch, derin vs sığ

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

**S: Custom CNN v2'de hangi yaratıcı özellikler var?**
C: 5 ana yenilik: (1) **Stem** (7x7 büyük kernel) - CT'de geniş alandan ilk bakışı yakalar, (2) **Multi-Scale Block** - 1x1, 3x3, 5x5 paralel yollarla hem küçük hem büyük kanamaları tespit eder, (3) **SE Attention** - hangi feature kanallarının önemli olduğunu öğrenir, (4) **Residual bağlantılar** - gradient flow iyileştirir, derin ağ eğitilebilir, (5) **Classifier'da LayerNorm** - son katmanda da stabilizasyon. Mimari: Stem(224→56) → MultiScale(56) → ResidualSE×3(56→28→14→7) → GAP → FC(256→128→2).

**S: SE Block ne işe yarıyor, neden eklediniz?**
C: Squeeze-and-Excitation (Hu et al., 2018) her feature kanalının önemini öğrenir. Medikal görüntüde bazı kanallar kenar, bazıları doku, bazıları yoğunluk bilgisi taşır. SE Block model'e "kanama tespiti için hangi bilgi türü kritik?" sorusuna cevap öğretir. Reduction=8 ile parametre maliyeti minimum.

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
