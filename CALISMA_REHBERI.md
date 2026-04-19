# BM 480 - PROJE 2 ÇALIŞMA REHBERİ
## Head CT Hemorrhage Classification (Binary Sınıflandırma)
### Terimler · Kararlar · Neden Kullanıldı · Hoca Soruları

---

## 0. PROJE UYUM TABLOSU (Şartname vs Proje)

| # | Şartnamede İstenen | Projede Karşılığı | Dosya |
|---|-------------------|-------------------|-------|
| 1 | Kaggle veri seti | 200 CT görüntü (100 Normal / 100 Hemorrhage) | `head_ct/`, `labels.csv` |
| 2 | Sınıflandırma problemi | **Binary classification** — Normal vs Hemorrhage | `config.py::CLASS_NAMES` |
| 3 | Ön işleme | Resize 224×224, RGB, train'den hesaplanan mean/std normalizasyonu | `src/data_preprocessing.py` |
| 4 | Veri artırımı | Flip, Rotation, ColorJitter, Affine, Blur, RandomErasing + Mixup + CutMix | `src/data_preprocessing.py`, `src/train.py` |
| 5 | 70/15/15 split | Stratified 70/15/15 → 139 train / 31 val / 30 test | `src/data_split.py` |
| 6 | 1 pretrained CNN | **ConvNeXt-Tiny** — ImageNet ağırlıkları, Progressive Unfreezing | `src/pretrained_model.py` |
| 7 | 1 özgün CNN | **Custom CNN v2** — Residual + CBAM + DropPath + DilatedDS (~1.45M param) | `src/custom_cnn.py` |
| 8 | Hiperparametre tuning | Grid Search: LR × batch × weight_decay (12 kombinasyon) | `src/hyperparameter_tuning.py` |
| 9 | Eğitim grafikleri | Loss & Accuracy eğrileri her iki model için | `results/*_training_curves.png` |
| 10 | Model kaydetme | `.pth` checkpoint (state_dict + hparams + epoch) | `models/` |
| 11 | Overfitting kontrolü | Generalization gap grafiği (kırmızı >%10, turuncu >%5) | `results/*_training_analysis.png` |
| 12 | Confusion Matrix + metrikler | Accuracy, Precision, Recall, F1 — 3 model için | `src/evaluate.py`, `results/` |
| 13 | Tahmin arayüzü | Gradio: görüntü yükle → tahmin + olasılık + Grad-CAM | `src/app.py` |
| 14 | IEEE rapor | Proje raporu | — |
| 15 | Akış şeması | Bölüm 1 | Bu dosya |
| 16 | 5× IEEE 2025 makale | Bölüm 9 | Bu dosya |

**Şartname ötesi ekstralar (profesyonellik):**
Ensemble · Grad-CAM · t-SNE · ROC-AUC + PR eğrileri · Mixup+CutMix · CBAM · DropPath · RandomErasing · External OOD testi

---

## 1. PROJENİN AKIŞ ŞEMASI

```
[Veri Seti: 200 Head CT Görüntüsü]
   100 Normal  +  100 Hemorrhage (dengeli)
         │
         ▼
[1. Stratified Split — Bölümleme]
   70% Train (139) | 15% Val (31) | 15% Test (30)
   Sınıf oranı her sette korunur (~%50/%50)
         │
         ▼
[2. Normalizasyon İstatistikleri]
   mean, std  ←  SADECE train seti (data leakage önlemi)
         │
         ▼
[3. Ön İşleme + Veri Artırımı]
   ┌─────────────────────────────────────────┐
   │  TRAIN SETI (augmentation AÇIK)         │
   │  Resize 224×224 → Flip → Rotation       │
   │  ColorJitter → Affine → GaussianBlur    │
   │  → ToTensor → Normalize → RandomErasing │
   │  Eğitim sırasında: Mixup veya CutMix    │
   └─────────────────────────────────────────┘
   ┌─────────────────────────────────────────┐
   │  VAL / TEST SETI (augmentation KAPALI)  │
   │  Resize 224×224 → ToTensor → Normalize  │
   └─────────────────────────────────────────┘
         │
         ▼
[4. Hiperparametre Tuning — Grid Search]
   LR × batch_size × weight_decay (12 kombinasyon)
   Val seti üzerinde değerlendirilir | Test ASLA kullanılmaz
         │
         ▼
[5. Model Eğitimi]
   ┌──────────────────────────────────────────────────────┐
   │  MODEL A: ConvNeXt-Tiny (Transfer Learning)          │
   │  Faz 1: Backbone dondur → Sadece head eğit (5 ep)   │
   │  Faz 2: Tüm ağ açık → Fine-tune (60 ep, LR=2e-5)   │
   │  Teknik: Mixup + Label Smoothing + Cosine LR         │
   └──────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────┐
   │  MODEL B: Custom CNN v2 (Sıfırdan Eğitim)           │
   │  Stem → MultiScale → ResidualCBAM×3 → DilatedDS     │
   │  100 epoch, batch=8, LR=3e-4, patience=25            │
   │  Teknik: Mixup+CutMix + RandomErasing + DropPath     │
   └──────────────────────────────────────────────────────┘
         │
         ▼
[6. Değerlendirme (Test Seti — 30 görüntü)]
   Confusion Matrix | Accuracy | Precision | Recall | F1
   ROC-AUC + PR Eğrisi | t-SNE | Grad-CAM
         │
         ▼
[7. Ensemble]
   ConvNeXt + Custom CNN → Soft Voting
   Optimal ağırlıklar val seti üzerinden bulunur
         │
         ▼
[8. Gradio Arayüzü]
   Görüntü yükle → Model seç → Tahmin + Olasılık + Grad-CAM
```

---

## 2. TERİMLER SÖZLÜĞÜ

### Veri İşleme Terimleri

| Terim | Açıklama | Projede Neden Kullanıldı |
|-------|----------|--------------------------|
| **Stratified Split** | Veriyi bölerken her sette sınıf oranını koru | 200 görüntü dengeli (50/50) → split sonrası da korunmalı |
| **Data Leakage** | Test bilgisinin eğitime sızması | Normalizasyon istatistikleri sadece train'den hesaplanır |
| **Data Augmentation** | Mevcut veriden yeni örnekler üretme | 200 görüntü DL için az; augmentation etkili örnek sayısını artırır |
| **Normalizasyon** | Piksel değerlerini standardize etme (mean=0, std=1) | Farklı CT protokollerinin parlaklık farklarını giderir |
| **RandomErasing** | Görüntünün rastgele bölgesini sıfırla (p=0.3) | Modelin belirli bir bölgeye ezberlemesini engeller; OOD dayanıklılığı |
| **Mixup** | İki görüntüyü λ oranında karıştır; etiketleri de karıştır | Karar sınırlarını yumuşatır, overconfident tahminleri önler |
| **CutMix** | Bir görüntüden bölge kes, diğerinden yapıştır | Spatial özellik öğrenimini zorlar; Mixup ile 50/50 dönüşümlü |

### Model Terimleri

#### ConvNeXt için

| Terim | Açıklama |
|-------|----------|
| **Transfer Learning** | ImageNet'te öğrenilmiş 28M ağırlığı başlangıç noktası olarak kullan |
| **Progressive Unfreezing** | Önce sadece classifier head eğit; sonra tüm backbone aç |
| **Fine-tuning** | Pretrained ağırlıkları yeni veriyle küçük LR ile güncelle |
| **Discriminative LR** | Backbone: 2e-5, Head: 2e-4 — backbone'u yavaş güncelle |
| **Catastrophic Forgetting** | Pretrained bilgiyi kaybetmek — Progressive Unfreezing bunu önler |

#### Custom CNN v2 için

| Terim | Açıklama |
|-------|----------|
| **Residual Connection** | Giriş + çıkış skip bağlantısı — gradient vanishing önler |
| **CBAM** | Kanal Attention + Uzamsal Attention — SE Block'tan güçlü (Woo et al. 2018) |
| **SE Block** | Sadece kanal bazlı attention — v1'de vardı, v2'de CBAM ile değiştirildi |
| **DropPath (Stochastic Depth)** | Eğitimde bloğun tamamını rastgele atla (0.00→0.15) |
| **Dilated Conv** | dilation=2 ile filtreyi seyrek uygula → geniş alıcı alan |
| **Depthwise Separable Conv** | Spatial + channel konvolüsyonunu ayır → ~10x az parametre |
| **Multi-Scale Block** | 1×1 + 3×3 + 5×5 paralel yollar → farklı boyutlardaki kanamaları yakala |
| **GELU** | Gaussian Error Linear Unit — ReLU'dan pürüzsüz, daha iyi gradyan |
| **Global Average Pooling** | Feature map'i tek vektöre indir — parametresiz, GAP → FC |
| **BatchNorm / LayerNorm** | Aktivasyonları normalize et — eğitimi hızlandırır |

### Eğitim Terimleri

| Terim | Açıklama |
|-------|----------|
| **AdamW** | Adam + decoupled weight decay — L2 reg'i doğru uygular |
| **Label Smoothing** | Hedef [1,0] → [0.95,0.05] — overconfident tahminleri cezalandır |
| **Cosine Annealing** | LR'yi kosinüs eğrisiyle azalt + warm restart — yerel minimumdan kaç |
| **Gradient Clipping** | Gradient normunu 1.0 ile sınırla — patlama önler |
| **Early Stopping** | Val loss iyileşmezse durdur — overfitting noktasını otomatik bul |
| **Weight Decay** | Büyük ağırlıkları cezalandır — L2 regularization |

### Değerlendirme Terimleri

| Terim | Açıklama |
|-------|----------|
| **Confusion Matrix** | TP / TN / FP / FN dağılımını görsel gösterir |
| **Precision** | "Kanama var" dediğinde ne kadar haklı? TP/(TP+FP) |
| **Recall (Sensitivity)** | Gerçek kanamaların ne kadarını yakaladı? **Medikal'de kritik** |
| **F1-Score** | Precision+Recall harmonik ortalaması |
| **ROC-AUC** | Threshold'dan bağımsız performans (0.5=rastgele, 1.0=mükemmel) |
| **Grad-CAM** | Modelin karar verirken baktığı bölge ısı haritası |
| **t-SNE** | 256-boyutlu feature'ları 2D'de göster — sınıf ayrımı kalitesi |
| **Ensemble** | ConvNeXt + Custom CNN soft voting — hatalar birbirini dengeler |

---

## 3. MODEL KARARLARI — NEDEN?

### MODEL A: ConvNeXt-Tiny

**Neden ConvNeXt (ResNet50 değil)?**
ConvNeXt-Tiny (Liu et al., 2022) ResNet ailesinin modernize edilmiş versiyonudur.
Farklar: depthwise 7×7 conv, inverted bottleneck, GELU, LayerNorm, daha büyük kernel.
Medikal görüntülerde ViT ile rekabet eden performans. `timm` kütüphanesi ile tek satır yükleme.
ResNet50 de çalışırdı fakat ConvNeXt daha güncel mimari blokları sayesinde daha iyi genelleme sunar.

**Neden Transfer Learning?**
200 görüntü ile 28M parametreyi sıfırdan öğretmek imkânsızdır.
ImageNet'te öğrenilmiş kenar, doku, renk geçişi gibi düşük seviye özellikler CT görüntülerine de uygulanabilir.
Biz sadece "bu özellikler hemorrhage mi, normal mi?" ayrımını öğretiyoruz.

**Neden Progressive Unfreezing?**
Backbone'u direkt açıp 200 görüntüyle fine-tune etmek → catastrophic forgetting.
Çözüm: Faz 1 (backbone dondurulmuş, sadece head, 5 epoch) ile head'in önce adaptasyonu sağlanır.
Faz 2 (tüm ağ, LR=2e-5) ile backbone yavaşça CT veriye adapte edilir.
Howard & Ruder (ULMFiT, 2018) bu yaklaşımı teorize etti.

---

### MODEL B: Custom CNN v2

**Neden sıfırdan eğitim (no pretrained)?**
Şartname "özgün CNN" istiyor. Custom CNN projenin orijinalliğini gösterir.
Pretrained ConvNeXt ile karşılaştırmak: transfer learning'in gücü vs sıfırdan öğrenme.

**Neden bu mimari?**

```
Stem (7×7, stride=2) → MultiScale (1×1+3×3+5×5) →
Block1: ResidualCBAM  48→ 64  (56→28)  drop_path=0.00
Block2: ResidualCBAM  64→128  (28→14)  drop_path=0.05
Block3: ResidualCBAM 128→256  (14→ 7)  drop_path=0.10
Block4: DilatedDS    256→256  (7→ 7)   drop_path=0.15
GAP → Linear(256→128) → LayerNorm → GELU → Dropout(0.5) → Linear(128→2)
```

- **Büyük Stem (7×7)**: CT görüntüsünün genel yapısını ilk adımda geniş bir alanla kavra
- **MultiScale**: Kanama küçük (subdural, nokta şeklinde) veya büyük (intracerebral) olabilir → 3 ölçek paralel
- **ResidualCBAM**: Skip connection (gradient flow) + CBAM (nereye bakılacağını öğren)
- **DilatedDSBlock**: Blok4'te geniş alıcı alan (11×11 efektif), az parametre
- **DropPath 0.00→0.15**: Derinleştikçe daha fazla drop → her blok bağımsız öğrensin

**200 görüntü için güçlü regularization:**
- DropPath (0.00→0.15) + Dropout2d (0.05→0.15) + Dropout(0.5) classifier'da
- Mixup (α=0.3) + CutMix (50/50) → etiket ve bölge karıştırma
- RandomErasing (p=0.3) → model spesifik piksele değil, genel örüntüye bakmalı
- Weight decay 5e-4 + Label smoothing 0.1
- Batch size 8: az örnek başına gradyan → gürültülü = regularizer

---

## 4. HOCA SORULARI VE CEVAPLAR

### Veri / Split Soruları

**S: Neden stratified split?**
C: 200 örnek küçük. Random split'te bir sınıf test setinde az temsil edilebilir. Stratified split her sette %50 Normal / %50 Hemorrhage oranını korur. Küçük veride bu ayrım kritik — 30 örnekli test setinde tek bir yanlış temsil tüm metriği bozabilir.

**S: Neden 70/15/15?**
C: Ders notunda küçük-dengeli veri seti için önerilen oran. Test: 30 örnek (her sınıftan 15), Val: 31 örnek. Bu sayılar model seçimi ve final test için yeterli. 80/10/10 da olabilirdi fakat 20 örneklik test seti yeterince güvenilir olmaz.

**S: Augmentation'ı split'ten önce uygulasaydınız ne olurdu?**
C: Aynı orijinal görüntünün augmented kopyaları hem train hem test setine düşebilirdi. Bu "data leakage"dır — model aslında test verisini dolaylı görmüş olur, performans yapay yükselir. Gerçek dünyada bu başarı elde edilemez. Kural: Split first, then preprocess.

**S: Mean/std neden sadece train'den hesaplanıyor?**
C: Test verisinin istatistiklerini eğitime dahil etmek data leakage'dır. Gerçek deployment'ta test verisi önceden bilinmez. Train seti "bilinen dünya"dır, normalizasyon parametreleri oradan çıkmalıdır.

**S: Augmentation sırası önemli mi?**
C: Evet. Önce geometrik dönüşümler (flip, affine, rotation) → sonra renk/yoğunluk (ColorJitter, Blur) → en son RandomErasing. Bu sıra semantik tutarlılık için: önce şekil boz, sonra renk varyasyonu ekle, en son bölge maskele. Tersi tutarsız örnekler üretir.

---

### ConvNeXt Soruları

**S: ConvNeXt neden diğer pretrained modellerden (ResNet, VGG) farklı?**
C: ConvNeXt (2022) "Modernizing a ConvNet" makalesiyle ResNet'i ViT ilkeleriyle güçlendirdi: depthwise 7×7 conv (büyük kernel), inverted bottleneck, GELU aktivasyon, LayerNorm. Daha geniş alıcı alan ve daha az inductive bias → daha iyi genelleme. ResNet50 veya VGG de çalışırdı ama ConvNeXt daha güncel standartı temsil ediyor.

**S: Progressive Unfreezing'in 2 fazı neden gerekli, neden 5 epoch?**
C: Faz 1 (5 ep, LR=1e-3, head only): ImageNet backbone'u bozma, classifier'ı CT sınıflarına adapte et. Faz 2 (60 ep, LR=2e-5, tüm ağ): Backbone hafifçe CT'ye uyarla. 5 epoch yeterli çünkü classifier başlangıçta çok kötü (random init) — backprop büyük gradyan üretirdi ve backbone bozulurdu. 5 epoch sonra classifier stabileşiyor. Çok az epoch (1-2) yetersiz; çok fazla (15+) backbone'u katılaştırır.

**S: Discriminative LR nedir, neden kullandınız?**
C: Backbone parametreleri LR=2e-5, head parametreleri LR=2e-4. Backbone zaten iyi eğitilmiş — küçük adımlarla ince ayar. Head ise CT sınıfları için yeni — daha büyük adımlarla hızlı öğrensin. Tek LR kullansaydık ya backbone çok hızlı değişirdi (catastrophic forgetting) ya da head çok yavaş öğrenirdi.

**S: "Pretrained ağırlıklar CT görüntülerine nasıl uygulanabilir? ImageNet vs CT çok farklı değil mi?"**
C: İyi soru. Düşük seviye özellikler (kenar, doku, renk geçişi, frekans patternleri) tüm görüntü türlerinde ortaktır. CT'de de aynı fiziksel özellikler var. Yüksek seviye özellikler (köpek kulağı vs kanama şekli) farklı — ama bunları fine-tuning aşamasında öğretiyoruz. ImageNet ağırlıkları "iyi bir başlangıç noktası" sağlıyor; sıfırdan başlamaktan çok daha verimli.

**S: ConvNeXt neden %100 test accuracy verdi?**
C: Transfer learning + 200 örnek dengeli + güçlü pretrained özellikler. Özellikle ConvNeXt'in ImageNet'te öğrendiği dokusal özellikler CT'de de ayırt edici. 30 örneklik test setinin de görece küçük olduğunu belirtmek gerekir — %100 "30/30 doğru" anlamına geliyor. İstatistiksel güven aralığı geniş (%82-100). Gerçek klinik doğrulama için çok daha büyük test seti gerekir.

---

### Custom CNN v2 Soruları

**S: Custom CNN'de en önemli bileşen ne?**
C: CBAM + Residual. Residual connection olmasa 5 blok derinliğinde gradient vanishing olurdu — model öğrenemezdi. CBAM olmasa model "nereye bakacağını" bilmez — CT'de kanama küçük bir bölgede, attention olmadan tüm görüntüye eşit bakılır. Bu ikisi birlikte "hem gradyan akışı hem lokalizasyon" sağlar.

**S: CBAM nedir, SE Block'tan farkı?**
C: SE Block (Squeeze-and-Excitation): Sadece kanal attention. GAP → FC → Sigmoid → her kanalı ölçekle. "Hangi özellik türü önemli?" sorusunu cevaplar. CBAM (Woo et al. 2018): İki aşama. (1) Channel Attention: avg+max pool → shared FC → sigmoid. (2) Spatial Attention: kanal boyunca avg+max → 7×7 conv → sigmoid → "nerede". SE sadece "ne", CBAM hem "ne" hem "nerede". CT'de kanamanın lokalizasyonu kritik → CBAM daha uygun.

**S: DropPath nedir, Dropout'tan farkı?**
C: Dropout: Tek tek nöronları/kanalları sıfırla. DropPath (Stochastic Depth, Huang et al. 2016): Eğitimde tüm residual bloğu atla → çıktı sadece shortcut (identity) olur. Kalan bloğun çıktısı `x/keep` ile ölçeklenir (inference'da bias yok). Bu projede kademeli: Block1=0.00, Block2=0.05, Block3=0.10, Block4=0.15. Derin bloklar daha fazla drop edilir çünkü düşük seviyeli özelliklerden bağımsız. Her blok başkasına güvenmeden öğrenmek zorunda → co-adaptation önlenir.

**S: Dilated Convolution ne yapar, neden Block4'te?**
C: Normal 3×3 conv: 3×3 pikseli görür. Dilation=2 ile 3×3 conv: iki pikselde bir örnekler → 5×5 alan görür ama sadece 9 parametre (boşluklu). Block4'te feature map 7×7 küçük. Bu küçük haritada dilation=2 ile efektif alıcı alan 11×11'e çıkar → diffüz veya büyük kanamaları global bağlamda kavrar. Depthwise-separable ile kombinasyonu parametre sayısını da minimal tutar (~68K, regular conv = ~589K).

**S: Multi-Scale Block neden önemli?**
C: CT'de kanama boyutu değişken: subdural hematom ince katman, intracerebral hematom büyük yuvarlak. 1×1 kernel: kanallar arası bilgi, nokta özellikler. 3×3 kernel: küçük yerel yapılar. 5×5 kernel (2 adet 3×3): daha büyük yapılar, 25 piksel efektif alan. Bu 3 ölçeği paralel alıp birleştirerek farklı boyuttaki kanamaları aynı anda yakalarız. Inception mimarisi (Szegedy et al. 2015) aynı fikri kullanır.

**S: Neden batch_size=8 seçildi?**
C: Grid Search'te Custom CNN için 8 optimal çıktı. 200 görüntü, 139 train. batch=8 ile her epoch ~17 adım. Küçük batch = gürültülü gradient = regularization etkisi. Büyük batch (32-64) ile gradient daha düzgün ama overfitting riski artar. Ayrıca küçük veride büyük batch her adımda çok benzer örnekler görür → çeşitlilik azalır.

**S: Neden 100 epoch ve patience=25?**
C: Custom CNN sıfırdan öğreniyor. Erken epoch'larda öğrenme yavaş. CosineAnnealing warm restart (T_0=10, T_mult=2) ile epoch 10, 20, 40, 80... noktalarında LR sıfırlanır → her restart yeni bir öğrenme fırsatı. patience=25: LR iki kez sıfırlanıp iyileşme yoksa dur. Çok küçük patience (5-7) LR yeniden artmadan önce durdurabilir.

---

### Eğitim Teknikleri Soruları

**S: Mixup formülü ve etiket hesabı nasıl?**
C: `x_mix = λ*x_a + (1-λ)*x_b`, `λ ~ Beta(α, α)`. Etiket: `loss = λ*CE(out, y_a) + (1-λ)*CE(out, y_b)`. α=0.3 seçildi: Beta(0.3,0.3) → uç değerlere yakın λ üretir → karışım çok hafif (orijinal görüntüye yakın). α=1.0 = uniform karışım = görüntüler eşit ağırlıklı karışır; CT için anlamsız (kanama %50 + normal %50 karışımı tıbbi anlam taşımaz).

**S: CutMix Mixup'tan neden iyi?**
C: Mixup'ta her piksel iki görüntünün karışımı → "hayalet" görüntü oluşur, doğal görünmez. CutMix (Yun et al. 2019): `x_a`'dan bir dikdörtgen keser, yerine `x_b`'nin aynı bölgesini yapıştırır. Görüntü doğal kalır, sadece bir bölge değişmiş. Etiket alan oranına göre dağıtılır: `y = area_ratio*y_b + (1-area_ratio)*y_a`. Spatial feature öğrenmeyi zorlar: model bölgeye bakmalı, tüm görüntüye değil.

**S: Label Smoothing formülü?**
C: `y_smooth = (1-ε)*y_onehot + ε/K`. K=2 sınıf, ε=0.1 → `[0.95, 0.05]`. Model "100% bu" diyemez. ε=0 standart CE; ε=0.1 ConvNeXt için yaygın; ε=0.2 çok agresif — model hiçbir şeyden emin olamaz. İkili sınıflandırmada overconfident tahminler genellikle test dışı verilerde kötü genelleşir.

**S: Cosine Annealing Warm Restart nedir?**
C: `η_t = η_min + 0.5*(η_max-η_min)*(1+cos(π*T_cur/T_max))`. T_0=10 → 10 epoch sonra LR η_max'a sıfırlanır. T_mult=2 → sonraki döngü 20 epoch, sonra 40. Warm restart: yerel minimumdan kaçmak için LR'yi yeniden artır. ReduceLROnPlateau sadece azaltır; CosineAnnealing restart ile birden fazla "keşif" şansı verir.

**S: AdamW ile Adam arasındaki fark?**
C: Klasik Adam'da weight decay L2 ile karıştırılır: gradient güncellemesinde momentum bozar. AdamW weight decay'i optimizer adımından ayırır: `w ← w - η*(g + λ*w)`. Bu "decoupled" form daha saf regularization sağlar. Loshchilov & Hutter (2019). Modern DL'de standart.

---

### Değerlendirme Soruları

**S: Medikal AI'da en kritik metrik hangisi?**
C: **Recall (Sensitivity)**. FN = kanamalı hastaya "normal" demek = müdahale gecikmesi = ölüm riski. FP = sağlıklıya "kanama var" demek = gereksiz tetkik, anksiyete. Tarama testlerinde FN maliyeti FP maliyetinden çok yüksek. Bu yüzden Recall ön planda; gerekirse threshold düşürülür (daha az FN ama daha çok FP).

**S: AUC=1.0 çıktı, şüpheli değil mi?**
C: 30 örnekli test setinde matematiksel olarak mümkün. ConvNeXt pretrained + dengeli veri + görsel olarak belirgin sınıflar (kanama/normal CT morfolojisi farklı). Ancak küçük test seti %95 CI geniştir (~%82-100). Gerçek klinik doğrulama için harici büyük test seti gerekir. Sunumda "30 örneklik test setinde" notu ile sunulmalı.

**S: Confusion Matrix'ten ne öğreniyoruz?**
C: TP: Doğru hemorrhage, TN: Doğru normal, FP: Yanlış alarm (normal'e kanama dedi), FN: Kaçırılan kanama. Medikal'de FN en tehlikelisi. Matrix'ten: model hangi sınıfta daha çok hata yapıyor? Sistematik bias var mı (hep bir sınıfa mı tahmin ediyor)?

**S: Grad-CAM ne söylüyor?**
C: Son conv katmanının gradyanlarından ağırlıklı feature map kombinasyonu. İyi model: kanama bölgesinde yüksek aktivasyon. Kötü model: CT kenarına, etikete, artifakta bakıyor (shortcut learning). CT'de kanama genellikle hiperdens (parlak) alan — model bunu lokalize edebiliyor mu? Grad-CAM görseli "evet" veya "hayır" sorusunu yanıtlar.

**S: t-SNE ne gösteriyor?**
C: Son FC katmanından önceki 256-boyutlu feature vektörleri 2D'ye indirgenir. İyi model: Normal ve Hemorrhage kümeleri 2D'de net ayrılmış. Kötü model: İç içe geçmiş kümeler. t-SNE görseli "model sınıfları temsil ediyor mu?" sorusunu görsel yanıtlar.

**S: Ensemble neden tek modelden iyi?**
C: İki farklı mimari farklı hata paterni gösterir. ConvNeXt (pretrained, büyük kernel, geniş bağlam) vs Custom CNN (sıfırdan, yerel özellikler, CBAM). Bir modelin yanlış tahmin ettiği örneği diğeri doğru tahmin edebilir. Soft voting: `p_final = w1*p_convnext + w2*p_custom`. w1, w2 validation seti üzerinden optimize edilir (test seti ASLA kullanılmaz).

---

### Genel / Etik Sorular

**S: Bu model klinik kullanıma hazır mı?**
C: Hayır. (1) 200 görüntü = çok az (klinik AI için binlerce gerekir), (2) Tek kaynak = domain shift riski, (3) FDA/CE klinik onay süreci geçilmeli, (4) Radyolog ile birlikte kullanılmalı. Bu proje akademik kanıt-of-concept; klinik araç değil.

**S: Model farklı hastaneden CT görüntüsü alırsa ne olur?**
C: Domain shift / Out-of-Distribution (OOD) problemi. Farklı CT cihazı → farklı HU (Hounsfield Unit) dağılımı → piksel istatistikleri farklı → model düşük performans gösterebilir. Bu projede `external_test/` klasörü farklı kaynak görüntülerle OOD testi için kullanıldı. Çözüm: domain adaptation, test-time augmentation veya daha çeşitli eğitim verisi.

**S: Yanlış sınıflandırma durumunda sorumluluk kime ait?**
C: Medikal AI etik ilkeleri: Son karar **doktora** aittir. AI karar destek aracıdır, karar aracı değil. Arayüzde "Tıbbi teşhis için kullanılamaz" uyarısı var. Yasal sorumluluk geliştirici şirket (CE/FDA mark), kullanıcı doktor arasında paylaşılır.

**S: Modelinizi sıkıştırmak gerekse (mobile deployment)?**
C: (1) Post-training quantization (INT8): `torch.quantize_dynamic` → 4× küçük, %1-2 acc kaybı. (2) Pruning: küçük ağırlıkları sıfırla → %50 sparse. (3) Knowledge Distillation: ConvNeXt → Custom CNN öğretmen-öğrenci. Custom CNN zaten 1.45M parametre ile hafif — quantization yeterli olur.

---

## 5. DOSYA YAPISI

```
D--2/
├── main.py                    ← Pipeline başlangıcı (--train / --eval / --app)
├── requirements.txt           ← Bağımlılıklar
├── CALISMA_REHBERI.md         ← Bu dosya
│
├── src/
│   ├── config.py              ← Yollar, sabitler, cihaz tespiti
│   ├── data_preprocessing.py  ← Dataset sınıfı, normalizasyon, transform pipeline
│   ├── data_split.py          ← Stratified 70/15/15 split
│   ├── data_augmentation.py   ← Augmentation görselleştirme (preview)
│   ├── custom_cnn.py          ← Custom CNN v2 mimarisi (CBAM+DropPath+DilatedDS)
│   ├── pretrained_model.py    ← ConvNeXt-Tiny yükleme ve freeze/unfreeze
│   ├── train.py               ← Eğitim döngüsü, Mixup+CutMix, plot_curves
│   ├── evaluate.py            ← Test metrikleri, Grad-CAM, t-SNE, ROC
│   ├── hyperparameter_tuning.py ← Grid Search
│   ├── web_crawler.py         ← Web-crawled görüntüler ile test
│   ├── gradcam.py             ← Grad-CAM ısı haritası
│   ├── visualizations.py      ← t-SNE, ROC-AUC, training analysis
│   └── app.py                 ← Gradio arayüzü
│
├── head_ct/head_ct/           ← 200 CT görüntüsü (000.png - 199.png)
├── labels.csv                 ← id, hemorrhage (0/1)
├── external_test/             ← OOD harici test (9N + 12H, farklı kaynak)
│   ├── normal/
│   └── hemorrhage/
├── test_image.py              ← Tek görüntü / klasör testi (4 model, renkli çıktı)
├── models/                    ← Kaydedilen modeller (.pth)
│   ├── convnext_tiny_best.pth
│   ├── custom_cnn_best.azveri.pth   ← Custom CNN — 200 görüntü ile eğitildi
│   ├── custom_cnn_best.cok.veri.pth ← Custom CNN — büyük veri seti ile eğitildi
│   └── train_stats.json       ← Normalizasyon mean/std
└── results/                   ← Tüm grafikler + özellik dosyaları
    ├── features_convnext.csv  ← ConvNeXt özellik vektörleri (200×768)
    └── features_custom_cnn.csv← Custom CNN özellik vektörleri (200×256)
```

---

## 6. ÇALIŞTIRMA KILAVUZU

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Tüm pipeline (augpreview + tune + train + eval + webcrawl)
python main.py

# Sadece model eğitimi
python main.py --train

# Sadece değerlendirme (eğitilmiş model gerekli)
python main.py --eval

# Gradio arayüzü
python main.py --app
# → http://localhost:7860

# Augmentation önizleme
python main.py --augpreview

# Web-crawled görüntülerde test
python main.py --webcrawl

# Tek görüntü veya klasör testi — 3 model + ensemble, renkli terminal çıktısı
python test_image.py goruntu.jpg
python test_image.py external_test/
python test_image.py                  # interaktif yol girişi
```

**Test sonuçları (external_test — 21 OOD görüntü):**
| Model | Accuracy | Normal Recall | Hemorrhage Recall |
|---|---|---|---|
| Ensemble | %95.2 | %100 | %91.7 |
| ConvNeXt | %90.5 | %100 | %83.3 |
| Custom CNN Çok Veri | %81.0 | %88.9 | %75.0 |
| Custom CNN Az Veri | %76.2 | %100 | %58.3 |

**Eğitim süresi (CPU):**
- ConvNeXt: ~15-30 dak (65 epoch, batch=16)
- Custom CNN: ~30-60 dak (100 epoch max, batch=8)

---

## 7. RAPOR GRAFİKLERİ

| Grafik Dosyası | Hangi Bölümde | Ne Gösterir |
|----------------|--------------|-------------|
| `augmentation_preview.png` | Preprocessing | 8 augmented örnek |
| `dataset_overview.png` | Veri | Sınıf dağılımı |
| `convnext_training_curves.png` | Training | ConvNeXt loss+acc eğrileri |
| `custom_cnn_training_curves.png` | Training | Custom CNN loss+acc eğrileri |
| `convnext_tiny_training_analysis.png` | Overfitting | Gap + LR schedule |
| `custom_cnn_training_analysis.png` | Overfitting | Gap + LR schedule |
| `convnext_tiny_confusion_matrix.png` | Sonuçlar | ConvNeXt CM |
| `custom_cnn_confusion_matrix.png` | Sonuçlar | Custom CNN CM |
| `model_comparison.png` | Sonuçlar | 3 model yan yana |
| `roc_auc_curves.png` | Sonuçlar | ROC + PR eğrileri |
| `convnext_tiny_tsne.png` | Analiz | ConvNeXt feature space |
| `custom_cnn_tsne.png` | Analiz | Custom CNN feature space |
| `convnext_tiny_gradcam.png` | Açıklanabilirlik | ConvNeXt ısı haritası |
| `custom_cnn_gradcam.png` | Açıklanabilirlik | Custom CNN ısı haritası |
| `convnext_grid_search.png` | HPO | ConvNeXt grid sonuçları |
| `custom_cnn_grid_search.png` | HPO | Custom CNN grid sonuçları |

---

## 8. PROJE FARKLILIKLARI (Sizi Öne Çıkaran)

1. **Grad-CAM**: Model nereye bakıyor? Medikal AI'da açıklanabilirlik zorunlu.
2. **Ensemble**: İki farklı mimari → hatalar dengelenir → tek modelden iyi.
3. **t-SNE**: Feature space görselleştirmesi — sınıf ayrımı kalitesi kanıtı.
4. **CBAM Attention**: SE Block'tan güçlü kanal+uzamsal dikkat. Kanama lokalizasyonu.
5. **DropPath (Stochastic Depth)**: Modern regularization — blok seviyesinde.
6. **CutMix + Mixup**: Spatial özellik öğrenimi + karar sınırı yumuşatma.
7. **Progressive Unfreezing + Discriminative LR**: ConvNeXt'i doğru fine-tune etme.
8. **Data Leakage Farkındalığı**: Split → normalize sırası bilinçli ve kurala uygun.
9. **Eğitim Analizi Grafiği**: Overfitting gap renk kodlu — %10 kırmızı, %5 turuncu.
10. **External OOD Test**: `external_test/` klasörü — farklı kaynak görüntülerinde test.

---

## 9. BENZERİ ÇALIŞMALAR (Literatür)

### Veri Setleri (Harici Test İçin)

| Kaynak | Boyut | Kullanım |
|--------|-------|---------|
| felipekitamura/head-ct-hemorrhage | 200 (bizim veri) | Eğitim |
| CQ500 (Qure.ai) | 491 scan | OOD test |
| RSNA 2019 ICH | 800K+ slice | Benchmark |
| Radiopaedia.org | Vaka bazlı | Sunum test görseli |

### IEEE Xplore 2025 Arama Stratejisi

```
ieeexplore.ieee.org → Advanced Search
"intracranial hemorrhage" AND "deep learning" AND "CT"  [Year: 2025]
"brain hemorrhage" AND "classification" AND "CNN"       [Year: 2025]
```

### Rapor Tartışma Cümlesi

"[1]'de ResNet-50 ile %93.4 accuracy elde edilirken, bizim ConvNeXt-Tiny tabanlı
yaklaşımımız 200 örneklik dengeli veri setinde %100 test accuracy ve %X F1
elde etmiştir. Fark, Progressive Unfreezing + Mixup kombinasyonunun küçük
veri setlerinde sağladığı regularization avantajına bağlanabilir [2]."

---

## 10. SUNUM HAZIRLIK

**15 dk Slayt Akışı:**
1. Problem (beyin kanaması, acil teşhis, AI desteği)
2. Veri seti (200 görüntü, dengeli, kaynak, kısıtlılık)
3. Pipeline şeması (Bölüm 1)
4. Preprocessing + Augmentation (neden bu sıra, data leakage)
5. **ConvNeXt**: Neden transfer learning? Progressive unfreezing nasıl?
6. **Custom CNN v2**: Mimari şeması, CBAM açıklaması, DropPath
7. Hiperparametre tuning (grid search sonucu)
8. Eğitim grafikleri (overfitting analizi)
9. Test sonuçları (CM + ROC + model karşılaştırma)
10. Grad-CAM görseli (model nereye bakıyor?)
11. Ensemble iyileştirmesi
12. Arayüz demo (canlı — gradio)
13. Kısıtlılıklar (200 görüntü, tek kaynak, OOD riski)
14. Etik uyarı + Gelecek çalışma

**Kontrol Listesi:**
- [ ] Her iki model `.pth` dosyası mevcut (`models/`)
- [ ] Tüm `results/` grafikleri oluştu
- [ ] `requirements.txt` güncel
- [ ] Arayüz farklı makinede test edildi
- [ ] Grad-CAM canlı çalışıyor (sunumda göster)
- [ ] 5 IEEE 2025 makale tablosu dolu (Bölüm 9)
- [ ] "Tıbbi teşhis için kullanılamaz" uyarısı arayüzde var
