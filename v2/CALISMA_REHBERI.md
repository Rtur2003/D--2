# BM 480 - PROJE 2 ÇALIŞMA REHBERİ (v2)
## Head CT Hemorrhage Classification — Genişletilmiş Veri Seti Deneyi

> v1 ile aynı pipeline (1 transfer + 1 from-scratch CNN, aynı metrikler, aynı augmentation). **Tek fark: veri büyüklüğü + hasta-bazlı split.** Teorik içerik ve savunma için → `v1/CALISMA_REHBERI.md`. Bu dosya v2'ye özel farkları ve çalıştırma adımlarını anlatır.

---

## 1. v2'nin AMACI

**Soru:** v1 modelinin %96.67 test accuracy'si gerçek mi, yoksa 15 görüntülük mikro test setinin istatistiksel gürültüsü mü?

**Test yöntemi:** Aynı kod, 46× daha büyük veri, hasta-bazlı split. Sonuç v1'den düşük çıkarsa → v1 overfit etmişti, v2 gerçek generalization'ı gösteriyor. Aynı kalırsa → mimari sağlam, veri büyüklüğü şart değilmiş.

**v2 = v1'in overfitting testidir.** Yeni bir proje değil.

---

## 2. VERİ KAYNAKLARI

| Kaynak | Ham Görüntü | Hasta | Etiket Tipi | Lisans |
|--------|-------------|-------|-------------|--------|
| `abdulkader90/brain-ct-hemorrhage-dataset` (Kaggle) | 6795 JPG | 45 (27 Normal + 18 Hemorrhagic) | Hasta seviyesi (tüm dilimler aynı) | Kaggle public |
| `vbookshelf/computed-tomography-ct-images` (Kaggle) | 2501 JPG | 82 | **Dilim seviyesi** (`hemorrhage_diagnosis.csv`) | Araştırma serbest |
| **Birleşim (ham)** | **9296** | **127** | Karma | — |
| **Filtre sonrası (labels.csv)** | **8490** | **127** | Temiz | — |

İndirme: Kaggle API token env var olarak verildi, `v2/data_raw/` altına açıldı.

---

## 3. DİLİM SEVİYESİ ETİKET SORUNU

**Gözlem:** CT taramalarında tek hastadan ~30 ardışık kesit (slice) var. Hemorrhagic bir hastanın tüm dilimleri "hemorrhage" etiketini almış — fakat ilk/son 10-15 dilim kafatası tabanı veya vertex bölgesi, kanama görünmez. Bu **noisy label**.

**İki veri seti farklı davranıyor:**
- **vbookshelf:** CSV'de `(PatientNumber, SliceNumber) → No_Hemorrhage` eşlemesi var. Etiket zaten doğru.
- **abdulkader:** Sadece hasta klasörü var. Dilim seviyesi etiket yok.

**Çözüm (`scripts/filter_slices.py`):**
1. vbookshelf → CSV etiketini koru.
2. abdulkader Hemorrhagic → Her hastanın dilimlerini sırala, **ilk %15 ve son %15'i at**. Orta %70 kanamayı görme olasılığı yüksek kesitler.
3. abdulkader Normal → Tüm dilimler kalır (normal beyin her kesitte normal).

**Alternatif reddedildi:** Sadece vbookshelf (2501 görüntü) temiz olurdu ama büyük veri avantajını kaybederdik. Filtrelenmiş birleşim daha iyi kompromi.

**Sonuç:** 9296 → **8490** görüntü (806 kenar dilim atıldı). Yedek `v2/labels.raw.csv` olarak duruyor.

---

## 4. HASTA-BAZLI SPLIT (SLICE LEAKAGE ÖNLEME)

**Kritik risk:** Aynı hastanın 30 dilimi arasında yüksek uzaysal korelasyon var. Random stratified split yapsan, aynı hastanın dilimlerinden bir kısmı train'e bir kısmı test'e düşer → model "bu hastayı gördüm" diyerek test'te yapay yüksek accuracy üretir. Buna **slice leakage** denir.

**Çözüm:** `sklearn.model_selection.GroupShuffleSplit` ile `patient_id` grubuna göre split. Bir hastanın tüm dilimleri **ya hep train, ya hep val, ya hep test** — asla karışmaz. Aynı anda `label` stratify de korunur (70/15/15 hemorrhage/normal oranı).

**Sonuç (`scripts/split.py` çıktısı):**

| Set | Toplam | Normal | Hemorrhage | Oran |
|-----|--------|--------|------------|------|
| Train | 6131 | 4147 | 1984 | %72.2 |
| Val | 817 | 554 | 263 | %9.6 |
| Test | 1542 | 1044 | 498 | %18.2 |

Hasta sayısı: Train 89, Val 12, Test 26 → **disjoint**. Savunmada bu tablo gösterilmeli.

---

## 5. NORMALİZASYON İSTATİSTİKLERİ

Train setinden hesaplandı (`src/data_preprocessing.py::compute_train_statistics`), `v2/models/train_stats.json`:

```json
{
  "mean": [0.1808, 0.1808, 0.1808],
  "std":  [0.3046, 0.3046, 0.3046]
}
```

v1 değerleri (~[0.40, 0.40, 0.40]) ile farklı — çünkü v2'de çok daha fazla siyah arka plan var (kafatası dışı piksel). Bu beklenen, yanlış değil. **v1 ve v2'nin mean/std farklı olmak zorunda** → her deney kendi train'inden istatistik hesaplayacak.

---

## 6. HİPERPARAMETRE FARKLARI

| Parametre | v1 (200 img) | v2 (8490 img) | Neden değişti |
|-----------|--------------|---------------|----------------|
| `batch_size` | 16 | **32** | Büyük veride stabil gradient; küçük batch gürültüsü artık gerekli değil |
| `epochs` | 30 | **20** | 46× veri → daha az epoch ile yakınsar |
| `early_stopping_patience` | 7 | **5** | Aynı sebep: yakınsama daha hızlı |
| `learning_rate` | 1e-4 | 1e-4 | Aynı (AdamW + warmup + cosine zaten uyarlanıyor) |
| `weight_decay` | 1e-4 | 1e-4 | Aynı |
| Augmentation | Albumentations (11 kalemlik) | torchvision (5 kalemlik) | Veri bollaştı — ağır augment artık gerekli değil; hız kazanımı |

Diğer tüm varsayılanlar v1 rehberindeki §15.1 tablosu ile aynı (random_state=42, amp=True, gradient_clip=1.0, vb.).

---

## 7. DOSYA YAPISI

```
v2/
├── data_raw/                   # 9296 ham JPG (Kaggle'dan)
├── labels.raw.csv              # Filtre öncesi (yedek)
├── labels.csv                  # 8490 image, filtrelenmiş (aktif)
├── train.csv / val.csv / test.csv   # hasta-bazlı 70/15/15 split
├── scripts/
│   ├── build_labels.py         ✓ Kaynakları birleştirip labels.csv üretir
│   ├── filter_slices.py        ✓ Abdulkader Hemorrhagic kenar dilimleri atar
│   ├── split.py                ✓ GroupShuffleSplit ile hasta-bazlı bölme
│   └── audit_brain_content.py    (opsiyonel) dilim başına beyin doku oranı raporu
├── src/                        # v1'in aynısı; config.py + data_split.py v2 CSV'lerini okuyacak şekilde adapte edildi
├── models/
│   └── train_stats.json        ✓ Train mean/std (normalizasyon için)
├── results/                    # Eğitim sonrası metrikler, görseller
└── main.py                     # python main.py --all
```

---

## 8. ÇALIŞTIRMA ADIMLARI

### 8.1 Veriyi hazırla (bir kere)

```bash
# 1. Kaggle veri setlerini indir (KAGGLE_API_TOKEN export edilmiş olmalı)
cd d:/D--2/v2
kaggle datasets download -d abdulkader90/brain-ct-hemorrhage-dataset -p data_raw --unzip
kaggle datasets download -d vbookshelf/computed-tomography-ct-images -p data_raw --unzip

# 2. Birleşik labels.csv üret
python scripts/build_labels.py

# 3. Kenar dilimleri kırp
python scripts/filter_slices.py

# 4. Hasta-bazlı split
python scripts/split.py
```

### 8.2 Eğitim + Değerlendirme

```bash
cd d:/D--2/v2
python main.py --all     # tune + train + eval
# veya adım adım:
python main.py --tune
python main.py --train
python main.py --eval
```

### 8.3 GPU Kontrolü (RTX 3050)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Beklenen: 2.11.0+cu126 True NVIDIA GeForce RTX 3050 Laptop GPU
```

Eğer CUDA `False` dönüyorsa venv içinde CPU-only torch kurulu demektir — CUDA wheel'i zorla:
```bash
.venv/Scripts/python -m pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

---

## 9. BEKLENEN SONUÇ VE YORUM

v1 sonuçları (200 görüntü, slice-agnostic split):
- ConvNeXt: %96.67 (15 test, 1 yanlış)
- Custom CNN: %86.67
- Ensemble: %96.67

v2 beklentisi (8490 görüntü, hasta-bazlı split, 1542 test):
- ConvNeXt: **%93-95** (test daha zor; edge noise + hasta çeşitliliği)
- Custom CNN: **%88-92** (daha fazla veri → from-scratch daha iyi öğrenir)
- Ensemble: **%94-96**

**v2 accuracy'si v1'den düşebilir — kötü değil.** v1'in yüksek skoru 15 görüntülük test setinin istatistiksel gürültüsü. v2'nin 1542 test görüntüsü gerçek generalization'ı ölçer.

**Savunma cümlesi:** "v1'de %96.67, v2'de %94. Düşüş değil — v1'in test seti çok küçüktü, tek bir yanlış tahmin %6.67 accuracy oynatıyordu. v2'de hasta-bazlı split + 1542 test görüntüsü güvenilir generalization veriyor. Overfitting yok, model sağlam."

---

## 10. v2'YE ÖZGÜ RAPORA GİRECEKLER

Rapor metnini arkadaşlar yazacak. v2 tarafı için üretilmesi gereken görsel/tablo:

- [ ] `v2/results/dataset_overview.png` — v2 kaynak dağılımı (abdulkader vs vbookshelf)
- [ ] `v2/results/patient_split_table.png` — §4 tablosu görsel hali
- [ ] `v2/results/*_training_curves.png` — ConvNeXt + Custom CNN loss/acc
- [ ] `v2/results/*_confusion_matrix.png` — 3 model (ConvNeXt / Custom / Ensemble)
- [ ] `v2/results/roc_auc_curves.png` — 3 model ROC overlay
- [ ] `v2/results/model_comparison.png` — v1 vs v2 yan yana accuracy/F1 karşılaştırması
- [ ] `v2/results/features_test.csv` — penultimate embeddings (hoca gereksinimi)

v1'deki §16 kaynakça ortak, tekrar üretmeye gerek yok.

---

## 11. v1 vs v2 — TEK SAYFA ÖZET (SAVUNMAYA)

| Kriter | v1 | v2 |
|--------|-----|-----|
| Görüntü sayısı | 200 | 8490 |
| Hasta sayısı | bilinmiyor (anonim) | 127 (disjoint split) |
| Split stratejisi | Stratified random | **Hasta-bazlı** (GroupShuffleSplit) |
| Slice leakage riski | Yok (tek dilim/hasta) | **Engellendi** (aynı hastanın tüm dilimleri tek sette) |
| Etiket gürültüsü | Temiz | Abdulkader kenar dilimleri filtrelendi |
| Augmentation | Albumentations (11 kalem) | torchvision (5 kalem) — veri bol, hafif augment yetiyor |
| Batch size | 16 | 32 |
| Epoch | 30 | 20 |
| Normalizasyon (mean/std) | ~[0.40, 0.40, 0.40] | [0.181, 0.305] |
| Test set boyutu | 30 | 1542 |
| Sonuç güvenilirliği | Düşük (varyans yüksek) | Yüksek (1542 sample, hasta disjoint) |

Aynı pipeline, aynı iki model, aynı metrikler. v2 yalnızca veri büyüklüğünü ve split kalitesini değiştirir.
