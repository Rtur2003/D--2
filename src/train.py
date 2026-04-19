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
#
# İLERİ TEKNİKLER:
# 1. Label Smoothing → Overconfident tahminleri önler
# 2. Mixup → Veri artırımı + regularization (iki görüntüyü karıştır)
# 3. Cosine Annealing → Daha yumuşak LR azaltma
# 4. Progressive Unfreezing → ConvNeXt için: önce head, sonra tüm ağ
# 5. Gradient Clipping → Gradient patlamasını önler
# =========================================================================

import copy
import json
import numpy as np
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, DEFAULT_HPARAMS,
    RANDOM_SEED
)
from data_split import get_split_data
from data_preprocessing import (
    HeadCTDataset, compute_train_statistics, get_transforms
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model, unfreeze_model


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
        num_workers=0, pin_memory=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    return train_loader, val_loader


# ── İLERİ TEKNİKLER ─────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: İki görüntüyü karıştırarak yeni eğitim örnekleri oluştur.

    Formül: x_mix = λ*x_i + (1-λ)*x_j
            y_mix = λ*y_i + (1-λ)*y_j

    Neden: Karar sınırlarını yumuşatır, overfitting azaltır.
    Referans: Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: iki hedefin ağırlıklı ortalaması."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: bir görüntünün dikdörtgen bölgesini başkasıyla değiştir.

    Mixup'tan farkı: piksel karışımı yerine bölge yapıştırma.
    Spatial feature öğrenmeyi zorlar — dağılım kaymasına karşı güçlü.
    Referans: Yun et al. (2019) "CutMix: Training Strategy..."
    """
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    y_a, y_b = y, y[idx]

    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)

    x_new = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_new, y_a, y_b, lam


# ── EĞİTİM FONKSİYONLARI ───────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """Tek bir epoch eğitim - mixup ve gradient clipping destekli."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and model.training:
            # 50% Mixup, 50% CutMix — her ikisi de spatial genellemeyi artırır
            if np.random.random() < 0.5:
                mixed_images, y_a, y_b, lam = mixup_data(
                    images, labels, mixup_alpha
                )
            else:
                mixed_images, y_a, y_b, lam = cutmix_data(
                    images, labels, mixup_alpha
                )
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_a).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        loss.backward()

        # Gradient clipping - gradient patlamasını önler
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

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
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    label_smoothing: float = 0.0,
    use_cosine: bool = False,
    optimizer: optim.Optimizer = None,
) -> Dict:
    """
    Model eğitimi - ileri tekniklerle.

    Yenilikler:
    - Label Smoothing: Overconfident tahminleri cezalandır
    - Mixup: Eğitim örneklerini karıştırarak regularization
    - Cosine Annealing: Warm restart ile LR schedule
    - Gradient Clipping: Gradient patlamasını önle
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS

    lr = hparams["learning_rate"]
    epochs = hparams["epochs"]
    patience = hparams["early_stopping_patience"]
    weight_decay = hparams["weight_decay"]

    model = model.to(DEVICE)

    # Label smoothing ile CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # Cosine Annealing veya ReduceLROnPlateau
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
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
    print(f"Label Smoothing: {label_smoothing}, Mixup: {use_mixup}, "
          f"Cosine: {use_cosine}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha, max_grad_norm=1.0
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        # LR Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if use_cosine:
            scheduler.step(epoch)
        else:
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

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  [BEST] Yeni en iyi model! Val Loss: {val_loss:.4f}")
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


def train_convnext_progressive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    hparams: Dict
) -> Dict:
    """
    ConvNeXt Progressive Unfreezing Eğitimi.

    Strateji (küçük veri için en iyi transfer learning yaklaşımı):
    Faz 1: Backbone dondur, sadece classifier head eğit (5 epoch)
           → Feature extractor bozulmasın, head görevini öğrensin
    Faz 2: Tüm ağı küçük lr ile fine-tune et
           → İnce ayar, backbone CT'ye adapte olsun

    Neden bu kadar önemli:
    → 200 görüntü ile 28M parametreyi direkt eğitmek = overfitting
    → Progressive yaklaşım ile model kademeli olarak adapte olur
    """
    print("\n" + "=" * 70)
    print("PROGRESSIVE TRAINING: ConvNeXt-Tiny")
    print("=" * 70)

    # ── FAZ 1: Backbone dondur, sadece head eğit ─────────────────────
    print("\n[FAZ 1] Backbone donduruldu - Sadece classifier eğitiliyor...")
    model = get_convnext_model(pretrained=True, freeze_backbone=True)
    model = model.to(DEVICE)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    phase1_epochs = 5
    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_mixup=False, max_grad_norm=1.0
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"  Faz1 Epoch [{epoch}/{phase1_epochs}] "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f}")

    # ── FAZ 2: Tüm ağı fine-tune et ─────────────────────────────────
    print("\n[FAZ 2] Tüm parametreler açıldı - Fine-tuning başlıyor...")
    unfreeze_model(model)

    finetuning_hparams = {
        **hparams,
        "learning_rate": 2e-5,
        "epochs": 60,          # 2 tam cosine cycle (10+20+30 ep)
        "early_stopping_patience": 15,  # restart sonrasi toparlanmaya izin ver
    }

    # Discriminative LR: backbone cok kucuk, head 10x buyuk
    backbone_params = [
        p for n, p in model.named_parameters() if "head" not in n
    ]
    head_params = [
        p for n, p in model.named_parameters() if "head" in n
    ]
    disc_optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params,     "lr": 2e-4},
    ], weight_decay=finetuning_hparams["weight_decay"])

    history = train_model(
        model, train_loader, val_loader,
        model_name="convnext_tiny",
        hparams=finetuning_hparams,
        use_mixup=True,
        label_smoothing=0.1,
        use_cosine=True,
        optimizer=disc_optimizer,
    )

    return history


def plot_training_curves(
    history: Dict,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Egitim grafiklerini ciz (rapor icin)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep_list = list(range(1, len(history["train_loss"]) + 1))

    # Cosine Annealing Warm Restart noktalarini tespit et
    restarts = []
    if "lr" in history and len(history["lr"]) > 1:
        for i in range(1, len(history["lr"])):
            if history["lr"][i] > history["lr"][i - 1] * 1.5:
                restarts.append(i + 1)   # 1-indexed epoch

    def _annotate_restarts(ax, y_top):
        for r in restarts:
            ax.axvline(r, color="#94a3b8", ls=":", lw=1.2)
            ax.text(r + 0.1, y_top * 0.92, "LR\nRestart",
                    fontsize=7, color="#64748b", va="top")

    # Loss grafigi
    ax = axes[0]
    ax.plot(ep_list, history["train_loss"], "b-o",
            label="Train Loss", lw=2, ms=4)
    ax.plot(ep_list, history["val_loss"], "r-o",
            label="Validation Loss", lw=2, ms=4)
    _annotate_restarts(ax, max(max(history["train_loss"]),
                               max(history["val_loss"])))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{model_name} - Loss Curves",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy grafigi
    ax = axes[1]
    ax.plot(ep_list, history["train_acc"], "b-o",
            label="Train Accuracy", lw=2, ms=4)
    ax.plot(ep_list, history["val_acc"], "r-o",
            label="Validation Accuracy", lw=2, ms=4)
    _annotate_restarts(ax, 1.0)

    # Mixup etkisi notu (val > train cogunlukta ise)
    val_gt = sum(v > t for v, t in
                 zip(history["val_acc"], history["train_acc"]))
    if val_gt > len(history["train_acc"]) * 0.6:
        ax.text(
            0.5, 0.03,
            "Not: Mixup augmentasyonu egitim dogrulugunu baskiyor"
            " — beklenen normal davranis",
            transform=ax.transAxes, fontsize=8.5, ha="center",
            color="#475569", style="italic",
            bbox=dict(fc="#f1f5f9", ec="#cbd5e1", pad=3, lw=0.8),
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{model_name} - Accuracy Curves",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.12])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Egitim grafikleri kaydedildi: {save_path}")
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
    mean, std = compute_train_statistics(train_paths)

    # İstatistikleri kaydet
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

    # ── 4. Model 1: ConvNeXt (Progressive Unfreezing) ─────────────────
    convnext_ckpt = MODELS_DIR / "convnext_tiny_best.pth"
    if convnext_ckpt.exists():
        print("\n" + "=" * 70)
        print("MODEL 1: ConvNeXt-Tiny -> Zaten egitilmis, atlaniyor.")
        print("=" * 70)
    else:
        convnext_history = train_convnext_progressive(
            train_loader, val_loader, DEFAULT_HPARAMS
        )
        plot_training_curves(
            convnext_history, "ConvNeXt-Tiny",
            save_path=str(RESULTS_DIR / "convnext_training_curves.png")
        )

    # ── 5. Model 2: Custom CNN (Mixup + Label Smoothing) ─────────────
    custom_ckpt = MODELS_DIR / "custom_cnn_best.pth"
    if custom_ckpt.exists():
        print("\n" + "=" * 70)
        print("MODEL 2: Custom CNN -> Zaten egitilmis, atlaniyor.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("MODEL 2: Custom CNN (Residual + SE + MultiScale)")
        print("=" * 70)

        custom_cnn = get_custom_cnn()

        custom_hparams = {
            **DEFAULT_HPARAMS,
            "learning_rate": 3e-4,
            "batch_size": 8,
            "weight_decay": 5e-4,   # Daha güçlü L2 — dağılım kaymasına karşı
            "epochs": 100,
            "early_stopping_patience": 25,
        }

        # Custom CNN icin batch_size degisti, DataLoader yeniden olustur
        train_loader_cnn, val_loader_cnn = create_dataloaders(
            train_paths, train_labels,
            val_paths, val_labels,
            mean, std,
            batch_size=custom_hparams["batch_size"],
            augment=augment
        )

        custom_history = train_model(
            custom_cnn, train_loader_cnn, val_loader_cnn,
            model_name="custom_cnn",
            hparams=custom_hparams,
            use_mixup=True,
            mixup_alpha=0.3,        # CutMix+Mixup karışımı için yeterli alpha
            label_smoothing=0.1,
            use_cosine=True,
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
