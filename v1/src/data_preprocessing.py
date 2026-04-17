# =========================================================================
# DATA PREPROCESSING (Veri Ön İşleme)
# =========================================================================
# - CSV'den etiketleri yükleme
# - Görüntü doğrulama ve kontrol
# - Normalizasyon istatistiklerini hesaplama (sadece train setinden)
# - PyTorch Dataset sınıfı
#
# KURAL: Preprocessing parametreleri (mean, std) SADECE train setinden
# hesaplanır → data leakage önlenir (Ders Notu Bölüm 5)
# =========================================================================

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_DIR, LABELS_CSV, IMG_SIZE, CLASS_NAMES


def load_labels() -> pd.DataFrame:
    """labels.csv dosyasını yükle ve doğrula."""
    df = pd.read_csv(LABELS_CSV)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"hemorrhage": "label"})

    # Dosya yollarını oluştur
    df["image_path"] = df["id"].apply(
        lambda x: str(DATA_DIR / f"{int(x):03d}.png")
    )

    # Dosya varlığını kontrol et
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"[WARNING] {len(missing)} görüntü bulunamadı!")

    print(f"[DATA] Toplam: {len(df)} görüntü")
    print(f"[DATA] Sınıf dağılımı: {dict(df['label'].value_counts())}")
    print(f"[DATA] {CLASS_NAMES[0]}: {(df['label']==0).sum()}, "
          f"{CLASS_NAMES[1]}: {(df['label']==1).sum()}")

    return df


def compute_train_statistics(image_paths: List[str]) -> Tuple[List[float], List[float]]:
    """
    SADECE train setindeki görüntülerden mean ve std hesapla.

    KRITIK: Bu istatistikler train seti dışındaki verilere uygulanır
    ancak SADECE train setinden öğrenilir → data leakage önlenir.
    """
    print("[PREPROCESS] Train setinden normalizasyon istatistikleri hesaplanıyor...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        total_pixels += arr.shape[0] * arr.shape[1]

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)

    print(f"[PREPROCESS] Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"[PREPROCESS] Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()


def get_transforms(
    mean: List[float],
    std: List[float],
    is_train: bool = False,
    augment: bool = False
) -> transforms.Compose:
    """
    Transform pipeline oluştur.

    KURAL: Augmentation SADECE train setine uygulanır.
    Validation ve Test setleri ham veri dağılımını temsil etmelidir.
    (Ders Notu Bölüm 2.2)
    """
    transform_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]

    if is_train and augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)


class HeadCTDataset(Dataset):
    """Head CT Hemorrhage Dataset."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


if __name__ == "__main__":
    df = load_labels()
    print("\nVeri seti başarıyla yüklendi.")
