# Head CT Hemorrhage Classification — Proje ve Çalışma Rehberi

Bu dosya, projeyi **hatasız ve adım adım** çalıştırabilmeniz için hazırlanmıştır.
Özellikle sınıflandırma ödevi için kritik noktalar önem derecesiyle işaretlenmiştir.

---

## 1) Projenin Kısa Özeti

- **Problem tipi:** İkili sınıflandırma (Binary Classification)
- **Sınıflar:** `Normal` ve `Hemorrhage`
- **Veri kaynağı (repo içi):**
  - Görüntüler: `./head_ct/head_ct`
  - Etiketler: `./labels.csv`
- **Ana giriş dosyası:** `./main.py`
- **Model yaklaşımı:**
  - Pretrained model: ConvNeXt-Tiny (`src/pretrained_model.py`)
  - Özgün model: Custom CNN (`src/custom_cnn.py`)

---

## 2) Repo Yapısı (Doğrulanmış)

```text
<project-root>
├── main.py
├── test_image.py
├── requirements.txt
├── labels.csv
├── CALISMA_REHBERI.md
├── download_models.py               # Release'ten model checkpoint indirme
├── head_ct/head_ct/                 # 200 görüntü
├── external_test/
│   ├── normal/
│   └── hemorrhage/
├── models/
│   └── train_stats.json
├── results/                         # Üretilen metrik/grafik dosyaları
└── src/
    ├── config.py
    ├── data_preprocessing.py
    ├── data_split.py
    ├── data_augmentation.py
    ├── pretrained_model.py
    ├── custom_cnn.py
    ├── train.py
    ├── evaluate.py
    ├── hyperparameter_tuning.py
    ├── visualizations.py
    ├── gradcam.py
    ├── app.py
    ├── web_crawler.py
    ├── download_web_samples.py
    ├── extract_features.py
    ├── threshold_analysis.py
    ├── diagrams.py
    └── cv_experiment.py
```

---

## 3) Önem Derecesi (Sınıflandırma Ödevi İçin)

### 🔴 KRİTİK (Önce bunlar doğru olmalı)

1. **Veri sızıntısı olmaması**
   - Split mantığı: `src/data_split.py`
   - Normalizasyon istatistikleri yalnızca train’den: `src/data_preprocessing.py`
2. **Model eğitim/değerlendirme ayrımı**
   - Eğitim: `src/train.py`
   - Final test değerlendirme: `src/evaluate.py`
3. **Doğru metriklerin raporlanması**
   - Accuracy, Precision, Recall, F1
   - Confusion Matrix çıktıları
4. **Tekrarlanabilirlik**
   - Seed ayarları: `train.py` / `hyperparameter_tuning.py`
5. **Çalıştırılabilir ana akışın bilinmesi**
   - Komutlar: `main.py` argümanları

### 🟠 YÜKSEK

1. Augmentation yaklaşımı ve gerekçesi
2. Hiperparametre araması (Grid Search) mantığı
3. ConvNeXt vs Custom CNN farklarının açıklanması
4. Grad-CAM / ROC / t-SNE görsellerinin yorumlanması

### 🟡 ORTA

1. Arayüz (Gradio) demosu
2. Web’den toplanan örneklerle ek test
3. Ek analiz script’leri (threshold, feature export vb.)

---

## 4) Kurulum

Proje Python bağımlılıklarını `requirements.txt` üzerinden bekler:

```bash
cd <project-root>
pip install -r requirements.txt
```

> Not: `torch`, `torchvision`, `timm`, `gradio` gibi paketler zorunludur.
>
> `models/` klasöründe checkpoint yoksa önce aşağıdaki komutu çalıştırabilirsiniz:
> `python download_models.py`
>
> Bu script, GitHub Release varlıklarından hazır `.pth` checkpoint dosyalarını indirir.
> Ödevde kendi eğitiminizi göstermek istiyorsanız `--train` ile üretim yapın; sadece demo/çalıştırma için hazır checkpoint indirilebilir.

---

## 5) Ana Çalıştırma Komutları

`main.py` içindeki komutlar:

```bash
cd <project-root>

# Tüm akış (varsayılan): augpreview + tune + train + eval + webcrawl
python main.py

# Sadece eğitim
python main.py --train

# Sadece değerlendirme
python main.py --eval

# Sadece hiperparametre araması
python main.py --tune

# Sadece augmentation önizleme
python main.py --augpreview

# Web-crawled test
python main.py --webcrawl

# Arayüz
python main.py --app
```

---

## 6) Ödev İçin Doğru Çalışma Sırası (Önerilen)

1. **🔴 Veri ve split doğrulama**
   - `labels.csv` ve görüntü yolları doğrulansın
2. **🔴 Train istatistikleri + preprocessing doğrulama**
3. **🟠 Hiperparametre araması (`--tune`)**
4. **🔴 Model eğitimi (`--train`)**
5. **🔴 Final değerlendirme (`--eval`)**
6. **🟠 Rapor görselleri ve metrik JSON dosyaları kontrolü**
7. **🟡 Arayüz demosu (`--app`)**

---

## 7) Beklenen Çıktılar

Çıktı klasörü: `./results`

Sık kullanılan dosyalar:
- `convnext_tiny_metrics.json`
- `custom_cnn_metrics.json`
- `convnext_tiny_confusion_matrix.png`
- `custom_cnn_confusion_matrix.png`
- `model_comparison.png`
- `roc_auc_curves.png`
- `convnext_tiny_gradcam.png`
- `custom_cnn_gradcam.png`
- `convnext_tiny_tsne.png`
- `custom_cnn_tsne.png`

Model/istatistik klasörü: `./models`
- `train_stats.json`
- Eğitim sonrası `.pth` checkpoint dosyaları

---

## 8) Tek Görüntü veya Klasör Testi

```bash
cd <project-root>

# tek dosya (repo içinden göreli yol)
python test_image.py ./external_test/normal/indir.jpg

# klasör (repo içinden göreli yol)
python test_image.py ./external_test
```

`test_image.py`, model checkpoint dosyalarını `models/` altında arar.
Checkpoint yoksa model yükleme adımı başarısız olur.
Bu durumda `python download_models.py` ile hazır checkpoint indirilebilir veya `python main.py --train` ile modeller yeniden eğitilebilir.

---

## 9) Sunum / Rapor İçin Kritik Kontrol Listesi

- [ ] 🔴 Veri sızıntısı olmadığını net anlat
- [ ] 🔴 Train/Val/Test ayrımını sayılarla ver
- [ ] 🔴 Accuracy + Precision + Recall + F1 birlikte sun
- [ ] 🔴 Confusion Matrix yorumunu sınıf bazlı yap
- [ ] 🟠 ConvNeXt ve Custom CNN’i karşılaştır
- [ ] 🟠 ROC/PR ve Grad-CAM görsellerini yorumla
- [ ] 🟡 Arayüzü canlı veya ekran görüntüsüyle göster

---

## 10) Sık Hatalar ve Çözümler

1. **`ModuleNotFoundError: torch`**
   - `pip install -r requirements.txt` çalıştırılmalı

2. **Checkpoint bulunamadı (`.pth`)**
   - Önce `--train` çalıştır
   - veya checkpoint dosyalarını `models/` içine koy

3. **`--eval` çalışırken dosya hatası**
   - `models/train_stats.json` ve ilgili model dosyaları mevcut olmalı

4. **Veri yolu hataları**
   - Repo kökünde çalıştığından emin ol:
     `<project-root>`

---

## 11) Kısa Sonuç

Bu proje, küçük ama dengeli bir CT veri setinde hem transfer learning (ConvNeXt) hem de özgün CNN yaklaşımını karşılaştıran uçtan uca bir sınıflandırma çalışmasıdır.

Ödev başarısını en çok etkileyen bölüm: **🔴 veri sızıntısını engelleyen doğru deney kurgusu + doğru metrik raporlama**.
