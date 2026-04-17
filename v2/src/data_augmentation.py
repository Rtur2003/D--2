# =========================================================================
# DATA AUGMENTATION (Veri Artırımı)
# =========================================================================
# KURALLAR (Ders Notlarından):
# 1. Augmentation SADECE train setine uygulanır (Bölüm 2.2)
# 2. Validation ve Test setleri ham veri dağılımını temsil etmelidir
# 3. Augmentation split'ten SONRA yapılmalı (Bölüm 6 - Golden Rules)
# 4. Aynı kaynaktan türetilmiş örnekler farklı setlere düşmemeli
# =========================================================================

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMG_SIZE


def get_train_augmentation() -> A.Compose:
    """
    Train seti için augmentation pipeline.

    Medikal görüntü augmentation stratejisi:
    - Geometrik dönüşümler (flip, rotation, shift)
    - Parlaklık/kontrast ayarları
    - Hafif blur ve noise
    - CLAHE (medikal görüntü için faydalı)
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),

        # Geometrik dönüşümler
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-20, 20),
            p=0.5
        ),

        # Renk/parlaklık ayarları
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Medikal görüntü için CLAHE
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Hafif blur
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # Hafif noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Elastic transform (medikal görüntüde yaygın)
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
    ])


def get_val_test_transform() -> A.Compose:
    """
    Validation ve Test seti için transform.
    SADECE resize - augmentation YOK.
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])


def augment_and_save_samples(
    image_paths: List[str],
    labels: List[int],
    output_dir: str,
    num_augmented_per_image: int = 3
) -> Tuple[List[str], List[int]]:
    """
    Train görüntülerini augment edip kaydet.
    Orijinal + augmented görüntülerin yollarını ve etiketlerini döner.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aug_transform = get_train_augmentation()
    all_paths = list(image_paths)
    all_labels = list(labels)

    print(f"[AUGMENT] {len(image_paths)} orijinal goruntu -> "
          f"{len(image_paths) * (1 + num_augmented_per_image)} toplam")

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        img = np.array(Image.open(img_path).convert("RGB"))

        for aug_idx in range(num_augmented_per_image):
            augmented = aug_transform(image=img)["image"]
            aug_img = Image.fromarray(augmented)

            aug_filename = f"aug_{idx:03d}_{aug_idx}.png"
            aug_filepath = output_path / aug_filename
            aug_img.save(str(aug_filepath))

            all_paths.append(str(aug_filepath))
            all_labels.append(label)

    print(f"[AUGMENT] Augmentation tamamlandı: {len(all_paths)} toplam görüntü")
    return all_paths, all_labels


def preview_augmentations(image_path: str, save_path: str = None) -> None:
    """Tek bir görüntü üzerinde augmentation örnekleri göster."""
    import matplotlib.pyplot as plt

    img = np.array(Image.open(image_path).convert("RGB"))
    aug_transform = get_train_augmentation()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Data Augmentation Örnekleri", fontsize=14, fontweight="bold")

    # Orijinal
    axes[0, 0].imshow(Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    axes[0, 0].set_title("Orijinal", fontweight="bold")
    axes[0, 0].axis("off")

    # Augmented versiyonlar
    for i in range(1, 8):
        row, col = divmod(i, 4)
        augmented = aug_transform(image=img)["image"]
        axes[row, col].imshow(augmented)
        axes[row, col].set_title(f"Augmented #{i}")
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[AUGMENT] Augmentation örnekleri kaydedildi: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    from config import DATA_DIR, RESULTS_DIR

    sample_img = str(DATA_DIR / "001.png")
    preview_augmentations(
        sample_img,
        save_path=str(RESULTS_DIR / "augmentation_preview.png")
    )
    print("Augmentation preview kaydedildi.")
