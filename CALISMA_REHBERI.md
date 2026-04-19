# BM 480 - PROJE 2 ÇALIŞMA REHBERİ
## Head CT Hemorrhage Classification
### Terimler, Kararlar, Neden Kullanıldı & Hoca Soruları

---

## 0. PROJE ŞARTNAMESİ UYUM KONTROLÜ (Proje2.docx)

Aşağıda hocanın dokümanda istediği her madde ve projede nasıl karşılandığı madde madde eşleştirilmiştir. Savunmada doğrudan bu tabloyu gösterebilirsiniz.

| # | Şartnamede İstenen | Projede Karşılığı | Dosya / Kanıt |
|---|-------------------|-------------------|---------------|
| 1 | Kaggle felipekitamura head-ct-hemorrhage veri seti | 200 görüntü (100N/100H) + **data_v2: ~6800 görüntü** (4105N/2690H) ek veri ile güçlendirme | `head_ct/`, `data_v2/`, `labels.csv` |
| 2 | Sınıflandırma problemi | Binary classification (Normal vs Hemorrhage) | `config.py::CLASS_NAMES` |
| 3 | Veri seti hazırlama (ön işleme / preprocessing) | Resize 224x224, RGB, train'den hesaplanan mean/std ile normalizasyon | `src/data_preprocessing.py` |
| 4 | Veri artırımı (data augmentation) | HorizontalFlip, VerticalFlip, Rotation(±20°), ColorJitter, Affine, GaussianBlur, AutoContrast, **RandomErasing** — yalnızca train'e. Eğitimde **Mixup + CutMix** (50/50) | `src/data_preprocessing.py` |
| 5 | Train + Validation + Test split | Stratified 70/15/15 → 139/31/30 (küçük veri); v2'de 70/15/15 + **WeightedRandomSampler** (sınıf dengesi) | `src/data_split.py`, `src/train.py` |
| 5a | Sunumda kullanmak için harici görüntüler | `external_test/` klasörü (Rumeysa'nın test seti: 9N+12H); OOD testi için | `external_test/` |
| 6 | 1 pretrained CNN (ConvNeXt, ResNeXt,...) | ConvNeXt-Tiny (timm), ImageNet ön-eğitimli, Progressive Unfreezing → **%100 test acc** | `src/pretrained_model.py`, `src/train.py` |
| 7 | 1 özgün CNN (kendi tasarımınız) | **Custom CNN v2**: Stem + MultiScale + 3×ResidualCBAM + DilatedDS + GAP + FC (≈1.37M param) | `src/custom_cnn.py` |
| 8 | Hyperparameter tuning (LR, batch size, epoch, early stopping patience, vs.) | Grid Search: LR × batch_size × weight_decay (12 kombinasyon) | `src/hyperparameter_tuning.py`, `results/*_grid_search.png` |
| 9 | Her iki CNN için train/val eğitim grafikleri | Loss & accuracy eğrileri + overfitting analizi | `results/convnext_training_curves.png`, `results/custom_cnn_training_curves.png` |
| 10 | Eğitilmiş modellerin saklanması | `.pth` checkpoint (model_state_dict + hparams + epoch + best_val) | `models/convnext_tiny_best.pth`, `models/custom_cnn_best.pth` |
| 11 | Overfitting trend kontrolü | Training analysis grafiği: gap eğrisi + renkli uyarı (>%10 kırmızı, >%5 turuncu) | `src/visualizations.py::plot_training_analysis` |
| 12 | Confusion Matrix, Accuracy, Precision, Recall | ConvNeXt + Custom CNN + Ensemble için ayrı ayrı | `src/evaluate.py`, `results/*_confusion_matrix.png` |
| 13 | Dosya seçimi ile tahmin eden arayüz + olasılık skoru | Gradio: görüntü yükle → 3 model seçimi → olasılık + Grad-CAM + rapor | `src/app.py` |
| 14 | IEEE Xplore makale formatında rapor | Proje raporu template'e göre | — |
| 15 | Akış şeması | Bölüm 1'de ve rapora eklenmek üzere | `CALISMA_REHBERI.md::1` |
| 16 | 5 adet IEEE Xplore 2025 makalesi | Bölüm 10'da arama stratejisi ve iskelet | `CALISMA_REHBERI.md::10` |
| 17 | Referanslar | IEEE stiliyle rapor sonu | Rapor |
| 18 | Kodlar çalışır durumda + requirements.txt | `main.py` tek komutla pipeline | `main.py`, `requirements.txt` |

**Öne çıkan EKSTRA (şartname istemedi ama profesyonellik için eklendi):**
- **Ensemble** (Soft Voting, ConvNeXt + Custom CNN)
- **Grad-CAM** açıklanabilirlik (hem evaluate.py'de hem arayüzde)
- **t-SNE** feature space görselleştirmesi
- **ROC-AUC + Precision-Recall** eğrileri
- **Mixup + CutMix + Label Smoothing + Cosine Annealing + Gradient Clipping + DropPath** (modern eğitim teknikleri)
- **CBAM (Channel + Spatial Attention)** — SE Block'tan üstün, kanama lokalizasyonu
- **RandomErasing + GaussianBlur** — dağılım kaymasına (OOD) dayanıklılık
- **WeightedRandomSampler** — v2 veri setindeki sınıf dengesizliğini (3:1) giderir
- **External test** — Farklı kaynak CT görüntülerinde OOD performans analizi
- **Profesyonel Arayüz** — Gradio 6 Sidebar + çok sekmeli sonuç paneli

---

## 1. PROJENİN AKIŞ ŞEMASI

```
[Veri Seti A] head_ct/ — 200 CT (100 Normal / 100 Hemorrhage)
[Veri Seti B] data_v2/ — ~6800 CT (4105 Normal / 2690 Hemorrhage)
[Harici Test]  external_test/ — 21 görüntü (Rumeysa; OOD testi)
         │
         ▼
[1. Ön İşleme (Preprocessing)]
    - Resize (224×224), RGB dönüşümü
    - Normalizasyon (mean/std → SADECE train'den hesaplanır)
         │
         ▼
[2. Veri Bölümleme (Data Split)]
    - Stratified Split: 70% Train / 15% Val / 15% Test
    - data_v2'de WeightedRandomSampler → sınıf dengesizliği (3:1) giderildi
    - random_state=42 (tekrarlanabilirlik)
         │
         ▼
[3. Veri Artırımı (Data Augmentation) — SADECE Train]
    - HorizontalFlip, VerticalFlip, Rotation(±20°)
    - ColorJitter, Affine, GaussianBlur, AutoContrast
    - RandomErasing (p=0.3) → OOD dayanıklılığı için bölge maskesi
    - Eğitimde: Mixup (50%) + CutMix (50%) → karar sınırı yumuşatma
         │
         ▼
[4. Model Mimarisi]
    ├── ConvNeXt-Tiny (Pre-trained, 28M param, Transfer Learning)
    └── Custom CNN v2 (~1.37M param, sıfırdan eğitim):
        Stem(7×7) → MultiScale(1×1+3×3+5×5) →
        ResidualCBAM×3 → DilatedDS(dil=2) → GAP → FC(256→128→2)
         │
         ▼
[5. Hiperparametre Tuning]
    - Grid Search: LR × Batch Size × Weight Decay (12 kombinasyon)
    - Validation seti üzerinde değerlendirilir (test ASLA kullanılmaz)
         │
         ▼
[6. Eğitim (Training) — İleri Teknikler]
    - AdamW + Gradient Clipping (max_norm=1.0)
    - Label Smoothing (0.05-0.1)
    - Mixup (α=0.3) + CutMix (β=1.0) dönüşümlü
    - Cosine Annealing (warm restart)
    - Progressive Unfreezing (ConvNeXt: head → backbone)
    - DropPath (Stochastic Depth, 0.00→0.15) — Custom CNN'de
    - Early Stopping (patience=7-25), Checkpoint (en iyi val_loss)
         │
         ▼
[7. Değerlendirme (Evaluation)]
    ├── Confusion Matrix (3 model: ConvNeXt / Custom CNN / Ensemble)
    ├── Accuracy, Precision, Recall, F1 (weighted + per-class)
    ├── ROC-AUC + Precision-Recall Eğrileri
    ├── t-SNE Feature Visualization
    ├── Grad-CAM (açıklanabilirlik)
    ├── Eğitim Dinamikleri Analizi (overfitting gap, LR schedule)
    ├── Ensemble Karşılaştırma (optimal ağırlık: validation'dan)
    └── OOD Testi: external_test/ üzerinde harici doğrulama
         │
         ▼
[8. Arayüz (Gradio)]
    - Dosya yükleme → Tahmin + Olasılık + Grad-CAM + Rapor
    - 3 model seçeneği (ConvNeXt / Custom CNN / Ensemble)
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
| **CBAM (Convolutional Block Attention Module)** | Kanal + Uzamsal dikkat mekanizması | SE Block'tan gelişmiş: hangi KANAL önemli + hangi KONUM önemli. CT'de kanama bölgesini lokalize eder. Woo et al. (2018) |
| **SE Block (Squeeze-and-Excitation)** | Kanal bazlı attention (CBAM'ın öncülü) | Her feature kanalının önemini öğrenir. v1'de kullanıldı, v2'de CBAM ile değiştirildi |
| **DropPath (Stochastic Depth)** | Eğitimde tüm bloğu rastgele atla (identity'e düşür) | Her blok bağımsız özellik öğrenir, co-adaptation önlenir. Derin bloklarda daha yüksek drop_prob (0.00→0.15) |
| **Dilated Convolution** | Filtreyi seyreltik uygula (dilation=2 → iki pikselde bir) | 7×7 feature map'te 11×11 efektif alıcı alan → büyük/diffüz kanamaları global bağlamda yakalar |
| **Depthwise Separable Conv** | Spatial (DW) + channel (PW) konvolüsyonunu ayır | Regular conv'ın ~10x az parametresi, daha iyi genelleme. Block4'te parametre verimli dilated işlem |
| **Multi-Scale Feature Fusion** | Farklı çözünürlüklerde (1x1, 3x3, 5x5) özellik çıkarma | CT'de kanama hem küçük (subdural) hem büyük (intracerebral) olabilir → farklı ölçeklerde bakmalıyız |
| **GELU Aktivasyon** | Gaussian Error Linear Unit — ReLU'dan pürüzsüz | Negatif değerleri yumuşakça bastırır (sert sıfırlama yok), daha iyi gradyan akışı |
| **BatchNorm / LayerNorm** | Aktivasyonları normalize etme | Eğitimi hızlandırır, internal covariate shift'i azaltır |
| **Global Average Pooling** | Feature map'leri tek bir vektöre indirger | Fully connected katmana göre çok daha az parametre → overfitting riski düşer |
| **Dropout / Dropout2d** | Eğitimde rastgele nöronları/kanalları devre dışı bırakma | Overfitting önlemi. Conv bloklarda %5-15, classifier'da %50 (v2'de artırıldı) |

### Eğitim Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **AdamW** | Adam optimizer + decoupled weight decay | Adam'ın regularization problemi çözülmüş hali, modern standart |
| **Label Smoothing** | Hedef etiketi [1,0] yerine [0.95, 0.05] yapma | Modelin %100 emin olmasını engeller → overconfident tahminleri önler, generalizasyonu artırır |
| **Mixup** | İki görüntüyü λ oranında karıştırarak yeni örnek üretme | x_mix = λ*x_i + (1-λ)*x_j. Karar sınırlarını yumuşatır, overfitting azaltır. Zhang et al. (2018) |
| **CutMix** | Bir görüntüden kesilen dikdörtgen bölgeyi diğerine yapıştırma | Mixup ile 50/50 dönüşümlü kullanılır. Spatial özellik öğrenimini zorlar; Yun et al. (2019) |
| **RandomErasing** | Görüntünün rastgele bir bölgesini sıfırla (p=0.3) | ToTensor sonrası uygulanır. Model bölge-bağımsız özellik öğrensin → OOD dayanıklılığı |
| **WeightedRandomSampler** | Sınıf frekansına ters orantılı örnekleme ağırlıkları | data_v2'deki 3:1 dengesizliği (4105N/2690H) gidermek için; her sınıf eşit olasılıkla seçilir |
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

### Neden Custom CNN v2 (~1.37M parametre)?
- 200 görüntü için 28M parametreli ConvNeXt bile çok büyük; Custom CNN daha az parametre = daha az overfitting riski
- Transfer learning ile ConvNeXt avantajlı başlar ama Custom CNN tamamen sıfırdan eğitiliyor; bu karşılaştırma projenin amacıdır
- **v2 Yenilikleri (v1 SE → v2 CBAM+DropPath+DilatedDS):**
  - **CBAM** (SE yerine): Hem kanal hem uzamsal attention → CT'de kanama lokalizasyonu daha hassas
  - **DropPath (Stochastic Depth)**: Blokları rastgele atla → co-adaptation önle, her blok bağımsız öğrensin
  - **DilatedDSBlock (Block4)**: dilation=2 ile 11×11 efektif alıcı alan, parametresiz (depthwise separable)
  - **GELU**: ReLU yerine pürüzsüz aktivasyon → daha iyi gradyan akışı
  - **CutMix + RandomErasing**: Uzamsal sağlamlık → OOD görüntülere daha dayanıklı
- **OOD testi sonucu**: data_v2 (~6800 görüntü) ile yeniden eğitim → farklı kaynak CT görüntülerinde genelleme hedefi

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

**S: CBAM nedir, SE Block'tan farkı nedir?**
C: **SE Block (Squeeze-and-Excitation)**: Sadece kanal bazlı attention — hangi feature kanalı önemli? **CBAM (Convolutional Block Attention Module, Woo et al. 2018)**: İki aşama. (1) **Channel Attention**: avg-pool + max-pool → shared FC → sigmoid → kanal ağırlıkları. (2) **Spatial Attention**: kanal boyunca avg+max → 7×7 conv → sigmoid → uzamsal ağırlık haritası. SE sadece "ne" sorusunu sorarken CBAM hem "ne" hem "nerede" sorusunu sorar. CT'de kanamanın lokalizasyonu kritik olduğundan CBAM v2'de tercih edildi.

**S: DropPath (Stochastic Depth) nedir, Dropout'tan farkı nedir?**
C: **Dropout**: Tek tek nöronları/kanalları sıfırlar. **DropPath (Huang et al. 2016)**: Eğitimde tüm residual bloğu rastgele atlar — blok sanki yokmuş gibi, çıktı identity (shortcut) olur. `keep = 1 - drop_prob`, geri kalan bloğun çıktısı `x / keep` ile ölçeklenir (inference zamanı bias yok). Her blok için drop_prob kademeli artar (0.00→0.05→0.10→0.15) — daha derin bloklar daha fazla drop edilir çünkü zaten düşük seviyeli özelliklere bağımlı değiller. Co-adaptation önler: her blok, diğerlerinden bağımsız özellik öğrenmek zorunda kalır.

**S: CutMix nedir, Mixup'tan farkı nedir?**
C: **Mixup**: `x_mix = λ*x_a + (1-λ)*x_b` — tüm görüntü piksel piksel karışır, yeni görüntü "hayalet" gibi görünür. **CutMix (Yun et al. 2019)**: `x_a`'dan rastgele bir dikdörtgen keser, yerine `x_b`'den aynı bölgeyi yapıştırır. Etiket: `y_mix = area_ratio * y_b + (1-area_ratio) * y_a`. Mixup'tan üstünlüğü: piksel düzeyinde anlamsız karışım yerine gerçek bölgeler kullanılır → spatial feature learning zorlanır. CT'de kanama bir bölgede lokal olduğundan CutMix modeli o bölgeye odaklanmaya iter. Bu projede Mixup ve CutMix 50/50 dönüşümlü uygulanır.

**S: WeightedRandomSampler ne yapar, neden gerekti?**
C: data_v2'de 4105 Normal / 2690 Hemorrhage (~3:2 ama Normal fazla). Eğer normal DataLoader kullansaydık her mini-batch ortalama %60 Normal, %40 Hemorrhage içerirdi → model Normal sınıfına biaslı öğrenirdi, Recall (Hemorrhage yakalama) düşük kalırdı. **WeightedRandomSampler**: Her örneğe `w = 1/class_freq` ağırlığı verilir. Her mini-batch'te her sınıftan eşit olasılıkla örnekleme yapılır. `replacement=True` çünkü azınlık sınıfı (`n_H=2690`) çoğunluktan az, tekrar örnekleme gerekir.

**S: OOD (Out-of-Distribution) problemi nedir, bu projede nasıl tezahür etti?**
C: **OOD (Dağılım Dışı)**: Model eğitim veri dağılımından farklı test verisiyle karşılaştığında performans düşer. Bu projede: orijinal head_ct (139 train, tek kaynak, Kaggle felipekitamura) ile eğitilen Custom CNN, Rumeysa'nın harici test görüntülerinde **%42.9 accuracy** elde etti (rastgeleden kötü). Neden? Farklı CT cihazı, farklı pencere/seviye ayarı, farklı hastane protokolü → piksel yoğunluk dağılımı farklı. Çözüm stratejileri: (1) **Daha büyük/çeşitli veri** (data_v2 ile yeniden eğitim), (2) **RandomErasing + CutMix** (bölge-bağımsız özellik), (3) **CBAM** (spatial lokalizasyon), (4) **TTA** (Test-Time Augmentation — çoklu tahmin ortalaması). Domain shift / data drift OOD'un alt terimleridir.

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

### Augmentation Detayı (Derin Sorular)

**S: Augmentation listesindeki sırayı neden bu şekilde seçtiniz? Sıra önemli mi?**
C: Evet, sıra önemlidir. Önce **geometrik dönüşümler** (flip, affine) uygulanır çünkü bunlar görüntünün "şeklini" değiştirir; sonra **renk/yoğunluk** (brightness, contrast, CLAHE) uygulanır. Böylece renk dönüşümleri geometrik olarak dönmüş görüntüye uygulanır. Blur ve noise en sonda, çünkü önceki augmentasyonlardan sonra eklenen bozunma en gerçekçi olur. Yanlış sıra (örn. önce blur, sonra flip) semantik olarak tutarsız örnekler üretir.

**S: Neden ElasticTransform kullandınız? Riskleri nelerdir?**
C: Elastic transform medikal görüntülerde organ/doku varyasyonunu simüle etmek için yaygındır (U-Net makalesinde de öneriliyor). Küçük `alpha=50, sigma=5` değerleriyle anatomik mantık korunur. **Risk:** Agresif elastic (alpha>200) kanama şeklini bozabilir → model yanlış özellik öğrenir. Bu yüzden `p=0.2` (düşük olasılık) ve küçük alpha seçildi.

**S: VerticalFlip medikal görüntü için doğru mu? Bir kafa ters durabilir mi?**
C: Gerçek hayatta kafa ters yatmaz, yani sembolik olarak VerticalFlip yanıltıcı olabilir. Ancak CT slice'ları farklı açılardan alınabilir ve radyolog görüntüleme yazılımında görüntüyü çevirebilir. Ayrıca CNN'in **rotation invariance** (dönme değişmezliği) özelliğini artırmak için VerticalFlip faydalıdır. Yine de `p=0.3` ile düşük olasılık seçildi — bu augmentasyon her zaman uygulanmıyor, sadece modele farklı perspektif öğretiyor.

**S: CLAHE parametrelerini (clip_limit=2.0, tile_grid_size=(8,8)) neden bu şekilde seçtiniz?**
C: `clip_limit=2.0`: Çok yüksek olursa (>4) noise artar. Çok düşük olursa (<1) CLAHE'nin etkisi kaybolur. 2.0 medikal görüntülerde literatürdeki yaygın değerdir. `tile_grid_size=(8,8)`: 224x224 görüntüyü 8x8 = 64 tile'a böler, her tile 28x28 piksel. Bu boyut kanama bölgesi kadar küçük alanların lokal kontrastını iyileştirir.

**S: Augmentasyon olasılıkları (p=0.5, 0.3, 0.2) nasıl seçildi?**
C: Her epoch'ta her görüntü farklı kombinasyonla gelir. Eğer tüm augmentasyonlar p=1.0 olsa, her örnek "deforme" olur ve orijinal dağılımı kaybederiz. p=0.5 en agresif (flip) için; p=0.3 orta (CLAHE, vertical flip); p=0.2 riskli olanlar (blur, noise, elastic) için. Bu sayede her örnek bazen "temiz" bazen "artmış" olarak gelir → model hem orijinal hem varyant dağılımı öğrenir.

**S: Mixup'ı neden `alpha=0.2` seçtiniz?**
C: Mixup formülü: `x_mix = λ*x_i + (1-λ)*x_j` ve `λ ~ Beta(α, α)`. **α=0.2** ise Beta dağılımı uç değerlere (λ≈0 veya λ≈1) yakın örnekler üretir → karışım çok "hafif" olur. α=1.0 ile uniform karışım (λ≈0.5) çok agresif ve medikal sınıflandırmada sınıflar arası semantik karışım anlamsız olur (kanama yok ile var %50/%50 karışmaz). α=0.2 orijinal Mixup makalesinde (Zhang et al., 2018) görüntü sınıflandırma için önerilen değerdir.

### Model Mimarisi Derin Sorular

**S: Custom CNN v2'nin toplam parametre sayısı nasıl hesaplandı?**
C: `torch.nn.Module`'ün `sum(p.numel() for p in model.parameters() if p.requires_grad)` ile. Custom CNN v2 için ≈1.3M; Kırılımı: Stem (≈1K), Multi-Scale (≈20K), 3x Residual-SE blok (≈400K+600K+200K), Classifier FC (≈50K). Detaylı rapor için `custom_cnn.py`'nin `__main__` bloğu parametre sayısını print eder.

**S: ConvNeXt-Tiny neden 28M parametreli?**
C: ConvNeXt ailesi büyüklüğe göre: Tiny (28M), Small (50M), Base (89M), Large (198M). Tiny seçildi çünkü: (1) 200 örnek için büyük model = overfitting, (2) CPU/GPU hesaplama yükü kabul edilebilir, (3) ImageNet üzerinde %82.1 top-1 accuracy ile yeterli pretrained güç var.

**S: Receptive field nedir, Custom CNN'de ne kadar?**
C: Receptive field, çıktı nöronunun giriş görüntüsünde "gördüğü" alan. Custom CNN v2: Stem(7x7) → MultiScale(5x5 dahil) → 3 conv blok (her biri 3x3). Toplam receptive field ≈ 7 + 4 + 2+4+8 ≈ 25+ piksel; ancak stride-2 pooling ile etkili receptive field tüm görüntüyü kapsar (224 piksel). ConvNeXt-Tiny'de daha büyük (sub-sampling + depthwise 7x7).

**S: Neden Multi-Scale Block'ta 1x1, 3x3, 5x5 kernel'ler seçildi, 7x7 niye yok?**
C: 1x1 (pointwise): kanallar arası bilgi; 3x3: yerel detay; 5x5: orta mesafe bağlam. 7x7 eklemek parametreyi ciddi artırır (49 > 25 > 9 > 1). Medikal CT'de kanama boyutları mm-cm arası, 5x5 (≈8 piksel spatial = birkaç mm) yeterli. Inception mimarisi de aynı kombinasyonu kullanır.

**S: SE Block'taki `reduction=8` nasıl çalışır?**
C: Kanal sayısı C ise → GAP ile C-boyutlu vektör → FC1: C/8 nöron (squeeze) → ReLU → FC2: C nöron (excitation) → Sigmoid → Orijinal feature map ile çarpılır. Reduction=8 parametre/performans optimum; reduction=16 daha küçük ama bilgi kaybı; reduction=4 daha güçlü ama parametreli. Orijinal SENet makalesinde de 16 varsayılan, biz küçük model için 8 seçtik.

**S: Global Average Pooling yerine Adaptive Average Pooling'i neden kullandınız?**
C: `nn.AdaptiveAvgPool2d(1)` input boyutundan bağımsızdır — 224x224 veya 256x256 gelsin, aynı çalışır. Klasik GAP (`nn.AvgPool2d(kernel)`) sabit kernel ister. Bu esneklik Grad-CAM'de 224'ten farklı boyutlarda da çalışmayı sağlar.

### Eğitim Teknikleri Derin Sorular

**S: Progressive Unfreezing'in fazlarını ve süresini neden öyle seçtiniz?**
C: **Faz 1 (head only, LR=1e-3, ~5 epoch)**: ConvNeXt backbone'un ImageNet ağırlıkları sabit, sadece classifier yeni verilere adapte olur. Hızlı kırılma yaşanmaz. **Faz 2 (all layers, LR=1e-4, 30 epoch)**: Backbone açılır, daha küçük LR ile fine-tune. Bu ayrım **catastrophic forgetting**'i önler (backbone'un önceden öğrendiği kenar/doku bilgisini kaybetmemesi için). Howard & Ruder (ULMFiT, 2018) bu stratejiyi ilk öneren makale.

**S: Mixup'ın doğruluk metriği nasıl hesaplanıyor? (karışık etiketler var)**
C: Mixup ile `y_mixed = λ*y_a + (1-λ)*y_b` olur. Loss hesabında `loss = λ*CE(output, y_a) + (1-λ)*CE(output, y_b)`. Ancak **accuracy** için: mixed batch'te hangi etiket "doğru" sayılır? Çözüm: `acc = λ * (pred==y_a).mean() + (1-λ) * (pred==y_b).mean()`. Bu projede `train.py`'nin eğitim döngüsünde bu düzeltme yapıldı (önceki bug fix'te eklendi).

**S: Cosine Annealing formülü nedir?**
C: `η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(πT_cur/T_max))`. Başlangıçta LR max, yarıda orta, sonda min (warm restart'ta sıfırlanır). **Warm Restart (SGDR)**: `T_max` dönemlerinde LR tekrar max'a atlar → yerel minimumdan kaçış. Bu projede `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`: ilk 10 epoch, sonra 20, 40...

**S: Gradient Clipping formülü ve etkisi?**
C: `if ||g||_2 > max_norm: g = g * (max_norm / ||g||_2)`. Gradient normu 1.0'ı aşarsa küçültülür. **Etki:** Patlayan gradient'i (özellikle küçük batch'te yaygın) engeller, eğitim stabilitesi sağlar. LSTM'de zorunludur, CNN'de bile modern best practice.

**S: Label Smoothing formülü nedir, neden değeri 0.05-0.1?**
C: `y_smooth = (1 - ε) * y_onehot + ε/K`. K=2 (sınıf sayısı), ε=0.1 → `[0.95, 0.05]`. **ε=0.1** ConvNeXt'te yaygın; **ε=0.05** Custom CNN'de kullanıldı (küçük veri setinde çok agresif smoothing performansı düşürebilir). ε=0 ise klasik CE; ε≥0.2 ise model çok "ürkek" olur, confident tahmin yapamaz.

**S: Weight decay ve L2 regularization arasındaki fark nedir?**
C: **L2 reg**: Loss'a `+ λ*||w||²` eklenir, gradient hesabında `∇L + 2λw` olur, Adam'da momentum'la etkileşir, yanlış ölçeklenir. **Weight decay (AdamW)**: Optimizer adımında direkt `w ← w - η*(g + λ*w)` uygulanır, momentum'dan bağımsız. AdamW'da weight decay doğru davranır. Bu yüzden `optim.AdamW(..., weight_decay=1e-4)` kullandık.

**S: Grid Search vs Random Search vs Bayesian?**
C: **Grid**: Tüm kombinasyonları dener, sistematik; pahalı (12 komb. × 20 epoch = 240 epoch). **Random**: Rastgele seçer, aynı süre içinde daha geniş arama; Bergstra-Bengio (2012) makalesinde grid'ten iyi çıktığı gösterildi. **Bayesian (Optuna/TPE)**: Önceki denemelerden öğrenerek bir sonrakini seçer; en verimli ama karmaşık kurulum. Bu proje için 12 kombinasyon yeterli → Grid tercih edildi.

**S: Reproducibility (tekrarlanabilirlik) nasıl sağlanıyor?**
C: `random_state=42` (split), `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.cuda.manual_seed(42)` + CuDNN benchmark kapalı. Ancak tam deterministic için `torch.use_deterministic_algorithms(True)` da gerekir (hız trade-off). Mixup + random augmentation sebebiyle tam aynı sonuç her run'da alınamaz ama trend benzer olur.

### Değerlendirme Derin Sorular

**S: Confusion Matrix'te FP ve FN'nin medikal anlamı nedir, hangisi daha ciddi?**
C: **FP (False Positive)**: Sağlıklı hastaya "kanaman var" demek → gereksiz ileri tetkik, anksiyete. **FN (False Negative)**: Kanaması olan hastaya "normal" demek → **müdahale gecikmesi, ölüm riski**. Medikal tarama modellerinde **recall (FN düşür)** genelde precision'dan (FP düşür) daha öncelikli. Bu projede her iki model de dengeli çalışıyor.

**S: Accuracy formülü nedir, dengeli veri setinde bile neden yetersiz?**
C: `Acc = (TP+TN)/(TP+TN+FP+FN)`. Dengeli veride bile tek bir sayı kaybın *nerede* olduğunu göstermez. Örneğin %90 accuracy: %10 kayıp FP mi FN mi? Bilemiyoruz → Confusion matrix + F1 + Recall gerekli.

**S: Precision, Recall, F1 formülleri?**
C: `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, `F1 = 2*P*R/(P+R)`. F1 harmonik ortalama çünkü aritmetik ortalama büyük değeri öne çıkarır, harmonik küçük değere ceza verir — yani precision=1 recall=0 olsa aritmetik 0.5 ama harmonik 0 → gerçek performans.

**S: AUC değeri 1.0 çıktı, bu güvenilir mi?**
C: **Dikkatli değerlendirme gerekir.** Dengeli 30 örnek test setinde AUC=1.0 matematiksel olarak mümkün (tüm threshold'larda iki sınıf doğru sıralanmış). ConvNeXt pretrained + 200 örnek + kolay ayrışabilen sınıflar (kanama vs normal görsel olarak belirgin) → olası. Ancak küçük test seti sebebiyle **generalization'ın garantisi değil**. Bu yüzden ensemble + external test (web-crawled) önerilir. AUC=1.0 sunarken "30 örneklik test setinde" kaydıyla söyleyin.

**S: ROC eğrisi ile Precision-Recall eğrisi arasındaki fark nedir?**
C: **ROC (TPR vs FPR)**: Threshold değiştikçe nasıl performans; dengeli veride iyi gösterge. **PR (Precision vs Recall)**: Dengesiz veride daha bilgi verici çünkü FPR düşük (TN çok), PR minority class'ta kaybı yakalar. Bu proje dengeli → her ikisi benzer bilgi verir, biz ikisini de çizdik.

**S: t-SNE'nin perplexity parametresini neden default bıraktınız?**
C: t-SNE'de `perplexity ≈ sqrt(n)` iyi bir başlangıç. 200 örnek için ≈14. scikit-learn default 30, küçük veri için 10-30 arası çalışır. Çok büyük perplexity (>50) lokal yapıyı kaybeder; çok küçük (<5) gürültülü olur. Sabit bıraktık çünkü görsel ayrım yeterince belirgin.

**S: Grad-CAM'i neden ConvNeXt'in son katmanından alıyorsunuz?**
C: Grad-CAM formülü: `L^c = ReLU(Σ_k α_k^c * A^k)`, α: sınıf-spesifik kanal önem katsayısı (global avg pool of gradients). Son conv katman (Custom CNN'de `conv_block4`, ConvNeXt'te `stages[-1]`) en yüksek semantik bilgiye sahip ama yeterli spatial çözünürlüğü tutar (7x7 veya 14x14). Daha erken katman → spatial yüksek ama semantik düşük; daha geç (GAP sonrası) → spatial bilgi kaybolmuş.

### Genel/Etik/Teorik Sorular

**S: Neden binary classification, neden 5-sınıf ICH subtype değil?**
C: Kaynak veri seti (felipekitamura) sadece binary etiketlere sahip. Subtype sınıflandırma için RSNA veya Hemorica gibi etiketli setler gerekli. Projenin scope'u binary ile sınırlandırıldı çünkü: (1) Veri seti imkanı, (2) 200 örnekle 5 sınıf = sınıf başına 40 örnek, öğrenme zor.

**S: Modeliniz yeni bir hastaneden gelen CT'de çalışır mı? Domain shift nedir?**
C: **Muhtemelen düşük performansla.** Domain shift: Eğitim ve test veri dağılımı farklı olması (farklı CT cihazları, protokoller, popülasyonlar). Bu proje tek kaynak veriyle eğitildi → deployment'ta **domain adaptation** (fine-tune with target domain) veya **domain generalization** (augmentation ile çeşitlilik) gerekir. Sunum için web-crawled test bu zayıflığı göstermek içindir.

**S: Eğer yanlış sınıflandırırsa kim sorumlu?**
C: Medikal AI etiği: Son karar **doktora aittir** (FDA rehberi, EU AI Act). AI "karar destek" aracıdır, "karar aracı" değildir. Bu projenin arayüzünde de "Tibbi teshis icin kullanilamaz" uyarısı var. Yasal sorumluluk geliştirici (CE mark), satan şirket ve kullanıcı doktor arasında paylaşılır.

**S: Modelin kararını doktora nasıl açıklarsınız?**
C: (1) **Olasılık skoru**: %85 hemorrhage confidence, (2) **Grad-CAM**: hangi bölgeye baktığı, (3) **Ensemble detay**: iki modelin hemfikir olup olmadığı. Doktor bu üç bilgiyi kendi muayene bulgularıyla çapraz kontrol eder.

**S: Batch size 8 vs 16 vs 32 seçimi performansı nasıl etkiler?**
C: **Küçük batch (8)**: Gradient daha gürültülü → regularizer etkisi, küçük veride iyi; daha yavaş hesap. **Büyük batch (32)**: Daha stabil gradient; fakat küçük veride her epoch'ta daha az güncelleme. Grid search Custom CNN için batch=8 optimal bulundu — 200 örnekli veri setinde "noisy gradient" aslında fayda sağlıyor.

**S: Learning rate 1e-3 mü 1e-4 mü kullandığınız modele göre neden değişiyor?**
C: **ConvNeXt**: Phase 1 (head only) 1e-3 — classifier sıfırdan, yüksek LR OK; Phase 2 (full) 1e-4 — pretrained ağırlıklara zarar vermemek için küçük. **Custom CNN**: Sıfırdan eğitim, ama grid search 1e-4'ü optimal buldu (1e-3'te loss dalgalanıyor, 1e-5'te yakınsama yavaş).

**S: Neden kayıp fonksiyonu olarak CrossEntropy, Focal Loss değil?**
C: **Focal Loss** (Lin et al., 2017) dengesiz veri (sınıf oranı 1:100+) için tasarlandı, "hard examples"a odaklanır. Bu proje 50/50 dengeli → focal gereksiz. CE + Label Smoothing zaten yeterli düzenlileştirme sağlıyor.

**S: Modelinizi compress etmek (quantization, pruning) gerekseydi nasıl yapardınız?**
C: (1) **Post-training quantization** (INT8): PyTorch `quantize_dynamic` → 4x küçük model, %1-2 accuracy kaybı. (2) **Pruning**: `torch.nn.utils.prune` ile küçük ağırlıkları sıfırla, %50 sparse → aynı accuracy mümkün. (3) **Knowledge distillation**: ConvNeXt'ten Custom CNN'e bilgi aktarımı. Şu an mobil deployment hedefi yok ama sorulursa bu cevap.

**S: Overfitting gap nedir, sizin modelinizde değer kaç?**
C: `gap = train_acc - val_acc` (her epoch için). **>%10 = overfitting**, %5-10 = hafif, <%5 = iyi. Bu projede: ConvNeXt max gap ~%21 ama son epoch %7.7 (early stopping kurtardı); Custom CNN max %10.34, son %-15.76 (val train'den iyi — Mixup'tan etkilenen train metriği). Final değerler kabul edilebilir.

**S: Early stopping olmasa ne olurdu?**
C: Train accuracy %100'e yaklaşırdı, val accuracy önce artıp sonra düşmeye başlardı (overfitting klasik örüntüsü). Model train verisini ezberlerdi, generalization kaybolurdu. Early stopping val_loss'un patience epoch boyunca iyileşmemesi durumunda eğitimi keser — **validation curve'ün minimumunu otomatik yakalar.**

**S: Val vs test farkı nedir, iki ayrı set gereksiz mi?**
C: **Val seti**: Hyperparameter seçimi, early stopping, model seçimi için kullanılır → "val'e overfit" olur (biz farkında olmadan val'e göre ayar yaparız). **Test seti**: Yalnızca *son kez* model performansını ölçmek için, ASLA karar için kullanılmaz. Bu ayrım olmadan bildirilen performans "optimistic bias" içerir. Bu projede test seti grid search'e bile girmedi.

**S: 30 örneklik test seti istatistiksel olarak anlamlı mı?**
C: **Sınırlı anlamlılık.** %95 güven aralığı geniş olur: %96.7 (29/30) accuracy için CI ≈ [%82, %99.8]. Daha güvenilir sonuç için k-fold CV veya external dataset (CQ500, RSNA) gerekir. Raporda bu kısıtlılık açıkça belirtilmeli.

**S: Proje zaman trade-off'ları: daha büyük model vs daha çok augmentation?**
C: **200 örnek** veri bottleneck'idir. Daha büyük model (ConvNeXt-Base 89M) → overfit. Daha çok augmentation (ör. stronger Mixup, CutMix) → daha iyi generalization. Yani **veri/regularization > kapasite**. Biz ConvNeXt-Tiny + Custom CNN (1.3M) + agresif regularization kombinasyonunu seçtik.

**S: Ensemble'da ConvNeXt ve Custom CNN ağırlıkları nasıl bulundu?**
C: `ensemble.py::find_optimal_weights` — validation seti üzerinde `w1 ∈ [0, 1]` arasında 0.05 adımlarla 21 ağırlık denenir, en yüksek val accuracy veren seçilir. Test seti kullanılmaz — test leakage'i önlemek için.

**S: "Grad-CAM model kenar piksellere bakıyor" durumu olursa ne dersiniz?**
C: Bu **shortcut learning** işaretidir. Model veri setindeki artifakttan öğreniyor demektir (örn. kanama görüntülerinde resim kenarında hastane damgası). Çözüm: (1) Veri temizliği — artifakt maskeleme, (2) Daha çeşitli veri, (3) Center cropping, (4) Augmentation ile kenar çeşitliliği.

**S: Modelin karar sınırını sayısal olarak nasıl ayarlarsınız (threshold tuning)?**
C: `sklearn.metrics.roc_curve`'den FPR/TPR alıp Youden's J statistic (`J = TPR - FPR`)'i maksimize eden threshold seçilir. Medikal için genelde threshold < 0.5 (recall öncelikli). Bu projede varsayılan 0.5 kullanıldı ama threshold optimizasyonu `evaluate.py`'de eklenebilir.

---

## 5. DOSYA YAPISI ve AÇIKLAMALAR

```
DÖ-2/
├── main.py                     ← Ana pipeline, tek komutla her şeyi çalıştır
├── colab_train.py              ← Google Colab GPU eğitim scripti (standalone)
├── requirements.txt            ← pip install -r requirements.txt
├── CALISMA_REHBERI.md          ← BU DOSYA (sınav çalışma notu)
│
├── src/                        ← Tüm kaynak kodlar
│   ├── config.py               ← Yollar, sabitler, hiperparametre defaults
│   ├── data_preprocessing.py   ← Veri yükleme, normalize, Dataset sınıfı
│   ├── data_split.py           ← Stratified train/val/test bölümleme
│   ├── data_augmentation.py    ← Augmentation pipeline (albumentations)
│   ├── custom_cnn.py           ← Custom CNN v2 (CBAM+DropPath+DilatedDS, ~1.37M param)
│   ├── pretrained_model.py     ← ConvNeXt-Tiny (timm ile yükleme)
│   ├── train.py                ← Eğitim döngüsü, Mixup+CutMix, run_training_v2
│   ├── evaluate.py             ← Test metrikleri + tüm görselleştirmeler
│   ├── hyperparameter_tuning.py← Grid Search (LR, batch, weight decay)
│   ├── ensemble.py             ← Soft Voting Ensemble + optimal ağırlık
│   ├── gradcam.py              ← Grad-CAM ısı haritası üretimi
│   ├── visualizations.py       ← t-SNE, ROC-AUC, eğitim analizi
│   ├── app.py                  ← Gradio arayüzü (tahmin + Grad-CAM)
│   └── web_crawler.py          ← Harici görüntüleri test etme
│
├── head_ct/head_ct/            ← 200 CT görüntüsü (000.png - 199.png)  [orijinal]
├── data_v2/                    ← ~6800 CT görüntüsü (büyük veri seti)
│   ├── normal/                 ←   4105 Normal görüntü
│   └── hemorrhage/             ←   2690 Hemorrhage görüntü
├── external_test/              ← OOD harici test seti (Rumeysa'nın fotoğrafları)
│   ├── normal/                 ←   9 Normal görüntü
│   └── hemorrhage/             ←   12 Hemorrhage görüntü
├── labels.csv                  ← head_ct etiketleri (id, hemorrhage: 0/1)
├── models/                     ← Eğitilmiş modeller (.pth)
├── results/                    ← Tüm grafikler ve metrikler
└── web_crawled_test/           ← Sunum için ek harici test görüntüleri
```

---

## 6. ÇALIŞTIRMA KILAVUZU

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 1a. (İlk kurulumda) Modelleri GitHub Releases'ten indir
python download_models.py

# 2. Tüm pipeline'ı çalıştır (augmentation + tuning + train + eval)
python main.py

# 3. Sadece eğitim (orijinal head_ct 200 görüntü)
python main.py --train

# 4. Büyük veri seti ile eğitim (data_v2 ~6800 görüntü, Custom CNN, CPU)
python main.py --train-v2

# 5. Sadece değerlendirme (eğitim sonrası)
python main.py --eval

# 6. Arayüzü başlat
python main.py --app
# Tarayıcıda http://localhost:7860 adresine git

# 7. Sunum için web-crawled görüntüleri test et
# Önce web_crawled_test/ klasörüne CT görüntüleri koy
python main.py --webcrawl

# 8. Google Colab ile GPU eğitimi (Drive'a data_v2/ ve head_ct/ yüklendikten sonra)
# Colab'da: colab_train.py hücrelerini sırayla çalıştır (Runtime → Run All)
# Drive yolu: /content/drive/MyDrive/Data/data_v2  ve  .../head_ct
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

---

## 9. BENZER VERİ SETLERİ (Eğitim/Test Genişletme İçin)

Proje küçük bir veri setiyle (200 görüntü) başladı. Aynı konuda (head CT hemorrhage) daha büyük veya ek veri arayışınız varsa şu halka açık veri setleri kullanılabilir. Web-crawling yerine / ile birlikte bu veri setlerinden örnek test görüntüleri çekilebilir.

### 9.1 Tavsiye Edilen Veri Setleri

| Veri Seti | Boyut | Lisans | Uygunluk | URL |
|-----------|-------|--------|----------|-----|
| **felipekitamura/head-ct-hemorrhage** (şu an kullandığımız) | 200 slice (100/100) | CC0 | Baseline | kaggle.com/datasets/felipekitamura/head-ct-hemorrhage |
| **CQ500 (Qure.ai)** | 491 scan, 193,317 slice; 205 ICH + 54 normal | CC BY-NC-SA 4.0 | En dengeli ek veri; binary için 205 ICH vs 54 normal (10:1 imbalance) | headctstudy.qure.ai/dataset |
| **RSNA 2019 Intracranial Hemorrhage Detection** | 874,035 slice, 25,272 exam; 5 ICH alt tipi | Non-commercial (research only) | En büyük; subtype sınıflandırması için; train/test'e yetiyor | kaggle.com/c/rsna-intracranial-hemorrhage-detection |
| **Hemorica** (2025) | 372 scan, 5 ICH alt tipi + segmentasyon mask'ları | Academic | Yüksek kaliteli etiketleme; tek hastane kaynaklı (domain shift riski) | arxiv.org/abs/2509.22993 |
| **BHX (Brain Hemorrhage Extended)** | CQ500 üzerine bounding-box etiketleri | PhysioNet | Lokalizasyon için; sınıflandırmada ek kullanılabilir | physionet.org/content/bhx-brain-bounding-box |
| **HemSeg-200** | 200 voxel-annotated scan | Academic | Segmentasyon odaklı ama classification için de kullanılabilir | arxiv.org/html/2405.14559 |

### 9.2 Kullanım Stratejileri

**A) Hiç değiştirmeden dışarıdan test (sunum için):**
- CQ500'den ~10 slice (5 normal + 5 hemorrhage) indirin, resize edip `web_crawled_test/` klasörüne koyun.
- `main.py --webcrawl` ile model bu "görülmemiş" veride nasıl performans veriyor kontrol edin.
- Bu, sunumda "external validation" başlığı altında çok etkilidir; domain shift'i de göstermiş olursunuz.

**B) Daha büyük veriyle re-train (proje kapsamını genişletme):**
- CQ500'den slice-level etiketli data çıkarıp felipekitamura ile birleştirin.
- **Dikkat**: Lisanslar farklı (CC0 vs CC BY-NC-SA) → rapora belirtin.
- Split stratejisi: source-stratified split — her veri setinden orantılı train/val/test.
- Normalizasyon istatistiklerini yeniden hesaplayın (farklı CT cihazları farklı dağılım).

**C) Multi-dataset generalization test:**
- `train` = felipekitamura (200), `test_external` = CQ500 subset.
- Beklenen: accuracy düşer (domain shift). Düşüş miktarı modelin generalization gücünü gösterir.
- Bu, sunumda "modelimiz gerçekten öğrenmiş mi, yoksa kaynak-spesifik mi?" sorusunun cevabı.

### 9.3 Veri İndirme Notları

- **RSNA** için Kaggle hesabı + challenge katılımı gerekiyor (veri >400GB). Küçük bir subset için AWS Open Data Registry (https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection/) daha pratik.
- **CQ500** doğrudan indirilebilir ama DICOM formatında — `pydicom` ile PNG'ye çevirmeniz gerekir.
- **Hemorica** NIFTI formatında; `nibabel` ile okuma, slice seçimi ve PNG export.
- **Copyright uyarısı**: Her veri seti lisansına rapor/kodda atıf verin; commercial veya non-commercial kullanım kurallarına dikkat.

### 9.4 Web-Crawling için Güvenilir Kaynaklar

Proje şartnamesinde "web-crawling ile birkaç görüntü örneği" istendi. Güvenilir akademik/klinik kaynaklar:
- **Radiopaedia.org**: Her vakanın detaylı açıklaması + lisanslı görüntüler (CC BY-NC-SA).
- **NLM Open-i**: NIH'ın açık medikal görüntü arama motoru.
- **Radiology Assistant (Radboud)**: Eğitim amaçlı vakalar.
- Rastgele Google görseli yerine bu kaynakları kullanın — sunumda "güvenilir akademik kaynak" kaydıyla.

---

## 10. IEEE XPLORE LİTERATÜR İSKELETİ (2025 Makaleler)

Şartnamede "Brain hemorrhage ile ilgili 5 adet SADECE ve SADECE IEEE Xplore makale (2025)" istendi.

### 10.1 Arama Stratejisi

IEEE Xplore (ieeexplore.ieee.org) → Advanced Search:
- **Arama sorguları:**
  - `"intracranial hemorrhage" AND "deep learning" AND "CT"` → Published Year: 2025
  - `"brain hemorrhage" AND "CNN" AND classification` → 2025
  - `"head CT" AND "transfer learning" AND hemorrhage` → 2025
- **Filtreler:** Year = 2025, Content Type = Conferences / Journals, Subject = Bioengineering / Medical Imaging

### 10.2 Literatür Özeti Tablosu (Rapor için)

Her makale için aşağıdaki tablo kolonlarını doldurun (raporda yer alacak):

| Kaynak | Veri Seti | Model | Accuracy / F1 / AUC | Katkı / Sınırlılık |
|--------|-----------|-------|---------------------|--------------------|
| [Makale 1, 2025] | ... | ... | ... | ... |
| [Makale 2, 2025] | ... | ... | ... | ... |
| [Makale 3, 2025] | ... | ... | ... | ... |
| [Makale 4, 2025] | ... | ... | ... | ... |
| [Makale 5, 2025] | ... | ... | ... | ... |

### 10.3 IEEE Referans Formatı Örneği

```
[1] A. Yazar, B. Yazar, ve C. Yazar, "Başlık," IEEE Trans. Medical Imaging,
    vol. XX, no. Y, ss. 1234-1245, 2025, doi: 10.1109/TMI.2025.xxxxxxx.
```

### 10.4 Kendi Projenizi Literatürle Karşılaştırma

Rapor tartışma bölümünde: "[1] ImageNet pretrained ResNet-50 ile X dataset'inde %95 accuracy raporlarken, bizim ConvNeXt-Tiny tabanlı yaklaşımımız 200 örneklik dengeli veri setinde %96.7 accuracy elde etmiştir; fark, [2]'de önerilen progressive unfreezing + Mixup kombinasyonunun küçük veride sağladığı regularization avantajına bağlanabilir."

---

## 11. HOCANIN "AYRINTIYA İNEN" SORULARI (ÖNCEDEN HAZIRLIK)

Bu sorular savunmada ince teknik detaylara girmek isteyen hocaları hedefler.

### 11.1 Kod/Mimari Ayrıntıları

**S: `train.py`'de en fazla hangi satırı savunabilirsiniz? Neden?**
C: Mixup accuracy hesabı düzeltmesi (lambda-weighted accuracy). Çünkü standart `(pred == target).mean()` Mixup'lı eğitimde *yanlış* sonuç verir — target batch'te biri mixup_a biri mixup_b ise hangisini doğru sayacağız? Biz her ikisini de λ/(1-λ) ağırlıklı topladık. Bu küçük fark bug olarak uzun süre gözden kaçabilir.

**S: `create_dataloaders` neden Custom CNN için ayrı çağrılıyor?**
C: Grid search Custom CNN için batch_size=8 optimal bulmuş, ConvNeXt için 16. Aynı DataLoader ile iki modeli eğitirsek ya Custom CNN suboptimal batch ile eğitilir ya da ConvNeXt. Bu yüzden `train_loader_cnn`, `val_loader_cnn` ayrı oluşturuluyor (`train.py`'de).

**S: `gradcam.py::get_target_layer` nasıl çalışıyor, model-agnostic mi?**
C: Model adına göre (`convnext`, `custom`) uygun son konvolüsyon bloğunu döner. Timm modelleri için `model.stages[-1]`, Custom CNN için `model.conv_block4`. Grad-CAM backward hook burada kaydedilir — gradient ve feature map tutulur, sonra ağırlıklı toplamla heatmap üretilir.

**S: `visualizations.py::plot_training_analysis` hangi grafikleri üretiyor?**
C: 2x2 subplot: (1) Train vs Val loss, (2) Train vs Val accuracy, (3) Gap eğrisi + overfitting uyarı renkleri, (4) LR schedule (varsa). Bu grafikle savunmada "model ezberlememiş" tezi görsel olarak ispatlanır.

### 11.2 "Ya Şu Olsaydı" Soruları

**S: Veri 200 değil 20 olsaydı ne yapardınız?**
C: (1) Transfer learning olmadan mümkün değil — pretrained zorunlu, (2) K-fold CV (stratified 5-fold), (3) Çok agresif augmentation + CutMix, (4) Few-shot learning yöntemleri (Prototypical Networks), (5) Binary yerine self-supervised pretraining + linear probing.

**S: Veri 20.000 olsaydı?**
C: (1) Daha büyük model (ConvNeXt-Base), (2) Scratch eğitim mümkün hale gelir, (3) K-fold gereksiz, tek split yeterli, (4) 90/5/5 split, (5) Subtype classification (5 sınıf) denenebilir.

**S: Class imbalance 9:1 olsaydı (bu projede 1:1)?**
C: (1) **Stratified split** zorunlu, (2) **Weighted CE loss** (`weight` tensörü = 1/freq), (3) **Focal Loss** hard examples'a odaklanır, (4) **SMOTE** sentetik azınlık örnekleri, (5) **Oversampling** minority class augmentation. Metriklerde macro-F1 + per-class recall ön planda.

**S: Model 3 sınıflı (Normal, Mild Hemorrhage, Severe Hemorrhage) olsaydı neyi değiştirirdiniz?**
C: (1) `CLASS_NAMES` ve `NUM_CLASSES=3`, (2) `CrossEntropyLoss` otomatik 3 sınıfı kaldırır, (3) Confusion matrix 3x3, (4) Grad-CAM sınıf seçimi (`target_class` parametresi), (5) Accuracy yerine macro-F1 ön planda, (6) Ordinal ilişki için alternatif loss (ordinal regression) değerlendirilebilir.

**S: Eğitim 2-3 gün sürecek olsa (ConvNeXt-Large + fold CV), ne yapardınız?**
C: (1) Mixed precision training (AMP) ile %50 süre azalması, (2) Gradient accumulation ile batch'i büyütme, (3) DataLoader num_workers artırma, (4) Checkpoint her fold sonunda — preemption'a karşı, (5) Weights & Biases veya TensorBoard ile uzaktan izleme.

### 11.3 Debug/Prod Soruları

**S: Modeliniz bazı görüntülerde yanlış tahmin yapıyor — nasıl debug edersiniz?**
C: (1) Yanlış tahminleri `evaluate.py`'dan filtrele (y_true != y_pred), (2) Her biri için Grad-CAM çıkar — model nereye bakıyor?, (3) Soft-probability değerleri (emin mi değil mi?), (4) Augmentation kaynaklı ise non-augmented inference dene, (5) Benzer yanlışlar arasında ortak artifakt var mı? (hastane stamp, kenar vs).

**S: Modelinizi production'a nasıl taşırsınız?**
C: (1) **Model serving**: `torch.jit.trace()` ile TorchScript export, (2) **Inference optimization**: INT8 quantization, (3) **API**: FastAPI + async endpoint, (4) **Container**: Docker + nvidia-runtime, (5) **Monitoring**: Prometheus metrics + alert (confidence dağılımı kayması = retrain sinyali), (6) **A/B test** + canary deployment.

**S: Model şimdi çalışıyor ama 3 ay sonra performansı düşebilir — neden, nasıl izlenir?**
C: **Data drift** (CT cihazı üreticisi yeni protokol yazdı) veya **concept drift** (yeni tip kanama görünümü) olabilir. İzleme: (1) Günlük confidence histogramı (belirsizlik artıyor mu?), (2) Prediction distribution shift, (3) Human-in-the-loop audit %5 örnek, (4) Shadow deployment yeni model + eskiyi karşılaştır.

---

## 12. SUNUM HAZIRLIK ÖZET

**Slayt Akışı Önerisi (15 dk):**
1. Problem (medikal AI, beyin kanaması acil teşhis)
2. Veri seti (200 görüntü, dengeli, sınırlılık)
3. Pipeline şeması (Bölüm 1)
4. Preprocessing + Augmentation (neden bu sıra?)
5. İki model mimari karşılaştırma (ConvNeXt vs Custom CNN)
6. Hyperparameter tuning (grid search tablosu)
7. Eğitim grafikleri (overfitting yok kanıtı)
8. Test metrikleri (CM + ROC + PR)
9. Grad-CAM görselleri (açıklanabilirlik)
10. Ensemble iyileştirmesi
11. Arayüz demo (canlı)
12. Literatür karşılaştırma (5 IEEE makale)
13. Kısıtlılıklar + gelecek çalışma
14. Etik + medikal uyarı

**İnteraktif demo ipucu:** Arayüzü sunum bilgisayarında açık tutun; hoca "şu görüntüyü dene" derse hemen gösterin. Grad-CAM canlı çıkarsa etki büyük.

**Sık unutulanlar:**
- [ ] requirements.txt güncel
- [ ] Modeller klasörünün GitHub'a yüklenmeyeceği (>100MB limit) — Google Drive linkini raporda paylaşın
- [ ] Akış şeması raporda yer alıyor
- [ ] Literatür tablosu 5 makale dolu
- [ ] Kod çalıştırılabilir (farklı makinede test et)
- [ ] README.md varsa güncel
