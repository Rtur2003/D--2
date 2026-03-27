# =========================================================================
# TRAINING (Eğitim)
# =========================================================================
# Her iki CNN modeli için eğitim pipeline'ı.
#
# KURALLAR (Ders Notlarından):
# - Early stopping, lr scheduling → validation setine göre (Bölüm 3.2)
# - Checkpoint seçimi → validation performansına göre (Bölüm 5)
# - Test seti eğitim sırasında KULLANILMAZ (Bölüm 7, Adım 5)
# - Eğitim grafikleri (train/val loss, accuracy) rapora eklenecek
# =========================================================================

import os
import copy
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, DEFAULT_HPARAMS,
    IMG_SIZE, RANDOM_SEED
)
from data_split import get_split_data
from data_preprocessing import (
    HeadCTDataset, compute_train_statistics, get_transforms
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Tekrarlanabilirlik için seed ayarla."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    train_paths, train_labels,
    val_paths, val_labels,
    mean, std,
    batch_size: int = 16,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Train ve Validation DataLoader oluştur."""

    train_transform = get_transforms(mean, std, is_train=True, augment=augment)
    val_transform = get_transforms(mean, std, is_train=False, augment=False)

    train_dataset = HeadCTDataset(train_paths, train_labels, train_transform)
    val_dataset = HeadCTDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Tek bir epoch eğitim."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validation değerlendirmesi."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    hparams: Dict = None,
) -> Dict:
    """
    Model eğitimi - early stopping ve checkpoint kaydetme ile.

    Early stopping ve checkpoint seçimi VALIDATION setine göre yapılır.
    (Ders Notu Bölüm 5 - Model Selection)
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS

    lr = hparams["learning_rate"]
    epochs = hparams["epochs"]
    patience = hparams["early_stopping_patience"]
    weight_decay = hparams["weight_decay"]
    sched_factor = hparams["scheduler_factor"]
    sched_patience = hparams["scheduler_patience"]

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor,
        patience=sched_patience
    )

    # Eğitim geçmişi
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"EĞİTİM BAŞLIYOR: {model_name}")
    print(f"{'='*60}")
    print(f"LR: {lr}, Batch: {hparams['batch_size']}, "
          f"Epochs: {epochs}, Patience: {patience}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        # LR Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Geçmişe kaydet
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Yazdır
        print(f"Epoch [{epoch:3d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # Early Stopping (validation loss'a göre)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ Yeni en iyi model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOP] {patience} epoch boyunca iyileşme yok. "
                      f"Eğitim durduruluyor.")
                break

    # En iyi modeli yükle ve kaydet
    model.load_state_dict(best_model_state)
    save_path = MODELS_DIR / f"{model_name}_best.pth"
    torch.save({
        "model_state_dict": best_model_state,
        "hparams": hparams,
        "best_val_loss": best_val_loss,
        "epoch": epoch
    }, str(save_path))
    print(f"\n[SAVE] En iyi model kaydedildi: {save_path}")

    # Geçmişi kaydet
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(str(history_path), "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_training_curves(
    history: Dict,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Eğitim grafiklerini çiz (rapor için)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss grafiği
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{model_name} - Loss Curves", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy grafiği
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(f"{model_name} - Accuracy Curves", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Eğitim grafikleri kaydedildi: {save_path}")
    else:
        plt.show()

    plt.close()


def run_training(augment: bool = True) -> None:
    """Ana eğitim pipeline'ı - her iki model için."""
    set_seed()

    # ── 1. Data Split ──────────────────────────────────────────────────
    train_df, val_df, test_df = get_split_data()

    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["label"].tolist()
    val_paths = val_df["image_path"].tolist()
    val_labels = val_df["label"].tolist()

    # ── 2. Train setinden normalizasyon istatistikleri ─────────────────
    # KURAL: Sadece train setinden hesapla (data leakage önleme)
    mean, std = compute_train_statistics(train_paths)

    # İstatistikleri kaydet (test ve arayüz için kullanılacak)
    stats = {"mean": mean, "std": std}
    stats_path = MODELS_DIR / "train_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)

    # ── 3. DataLoader'lar ──────────────────────────────────────────────
    batch_size = DEFAULT_HPARAMS["batch_size"]

    train_loader, val_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        mean, std,
        batch_size=batch_size,
        augment=augment
    )

    print(f"\n[DATA] Train: {len(train_loader.dataset)} örnek")
    print(f"[DATA] Validation: {len(val_loader.dataset)} örnek")
    print(f"[DATA] Augmentation: {'Aktif' if augment else 'Kapalı'}")

    # ── 4. Model 1: ConvNeXt (Pre-trained) ─────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL 1: ConvNeXt-Tiny (Pre-trained, Transfer Learning)")
    print("=" * 70)

    convnext = get_convnext_model(pretrained=True)
    convnext_history = train_model(
        convnext, train_loader, val_loader,
        model_name="convnext_tiny",
        hparams=DEFAULT_HPARAMS
    )
    plot_training_curves(
        convnext_history, "ConvNeXt-Tiny",
        save_path=str(RESULTS_DIR / "convnext_training_curves.png")
    )

    # ── 5. Model 2: Custom CNN ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL 2: Custom CNN (Özgün Mimari)")
    print("=" * 70)

    custom_cnn = get_custom_cnn()
    custom_history = train_model(
        custom_cnn, train_loader, val_loader,
        model_name="custom_cnn",
        hparams=DEFAULT_HPARAMS
    )
    plot_training_curves(
        custom_history, "Custom CNN",
        save_path=str(RESULTS_DIR / "custom_cnn_training_curves.png")
    )

    print("\n" + "=" * 70)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"Modeller: {MODELS_DIR}")
    print(f"Grafikler: {RESULTS_DIR}")


if __name__ == "__main__":
    run_training(augment=True)
