# =========================================================================
# ADVANCED VISUALIZATIONS (İleri Görselleştirmeler)
# =========================================================================
# 1. t-SNE Feature Space - Modelin öğrendiği temsilleri 2D'de göster
# 2. ROC-AUC Eğrileri - Medikal projede standart metrik
# 3. Precision-Recall Eğrileri - Dengesiz veriler için kritik
# 4. Eğitim dinamikleri analizi - Overfitting tespiti
#
# Bu görselleştirmeler projeyi akademik seviyeye taşır.
# =========================================================================

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE
)


# ── t-SNE Feature Extraction ───────────────────────────────────────────

class FeatureExtractor:
    """
    Modelin son katman öncesi feature'larını çıkar.

    Neden t-SNE?
    → Modelin yüksek boyutlu feature space'inde sınıflar ne kadar
      iyi ayrılmış? Bu görsel olarak gösterir.
    → İyi eğitilmiş model: 2 net küme (Normal vs Hemorrhage)
    → Kötü model: karışık noktalar
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.features = []

        # Hook: son FC katmanından önceki feature'ları yakala
        self._register_hook(model_name)

    def _register_hook(self, model_name: str):
        if "convnext" in model_name.lower():
            # ConvNeXt: head.global_pool sonrası
            target = self.model.head.global_pool
        else:
            # Custom CNN: global_pool sonrası
            target = self.model.global_pool

        def hook_fn(module, input, output):
            feat = output.view(output.size(0), -1)
            self.features.append(feat.detach().cpu().numpy())

        target.register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Tüm veri setinden feature çıkar."""
        self.features = []
        all_labels = []

        for images, labels in loader:
            images = images.to(DEVICE)
            _ = self.model(images)
            all_labels.extend(labels.numpy())

        features = np.vstack(self.features)
        labels = np.array(all_labels)

        return features, labels


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: str = None,
    perplexity: int = 15
) -> None:
    """
    t-SNE ile feature space görselleştirme.

    t-SNE (t-distributed Stochastic Neighbor Embedding):
    → Yüksek boyutlu veriyi 2D'ye indirger
    → Benzer örnekleri yakın, farklıları uzak tutar
    → Kümeleme kalitesini görsel olarak değerlendirir
    """
    print(f"[t-SNE] {model_name} - {features.shape[0]} örnek, "
          f"{features.shape[1]} boyut -> 2D")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(features) - 1),
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca"
    )
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"Normal": "#3b82f6", "Hemorrhage": "#ef4444"}
    markers = {"Normal": "o", "Hemorrhage": "^"}

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=colors[cls_name],
            marker=markers[cls_name],
            s=80, alpha=0.7,
            edgecolors="white", linewidth=0.5,
            label=f"{cls_name} (n={mask.sum()})"
        )

    ax.set_title(
        f"t-SNE Feature Space - {model_name}\n"
        f"İyi ayrılmış kümeler = model sınıfları iyi öğrenmiş",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("t-SNE Boyut 1", fontsize=11)
    ax.set_ylabel("t-SNE Boyut 2", fontsize=11)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[t-SNE] Kaydedildi: {save_path}")

    plt.close()


# ── ROC-AUC Eğrisi ─────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = None
) -> None:
    """
    ROC-AUC eğrileri - her iki model karşılaştırmalı.

    ROC (Receiver Operating Characteristic):
    → X ekseni: False Positive Rate (yanlış alarm)
    → Y ekseni: True Positive Rate (doğru tespit)
    → AUC: Eğri altı alan (1.0 = mükemmel, 0.5 = rastgele)

    Medikal AI'da neden önemli?
    → Threshold'u değiştirerek sensitivity/specificity dengesini gösterir
    → Kanama tespitinde FN (kaçırılan kanama) çok tehlikeli
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = ["#3b82f6", "#ef4444", "#10b981"]

    # Sol: ROC eğrileri
    ax = axes[0]
    for i, (name, (y_true, y_prob)) in enumerate(results.items()):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Rastgele (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (Yanlış Alarm)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Doğru Tespit)", fontsize=12)
    ax.set_title("ROC Eğrisi Karşılaştırması", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Sağ: Precision-Recall eğrileri
    ax = axes[1]
    for i, (name, (y_true, y_prob)) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, color=colors[i], lw=2.5,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall (Duyarlılık)", fontsize=12)
    ax.set_ylabel("Precision (Kesinlik)", fontsize=12)
    ax.set_title("Precision-Recall Eğrisi", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ROC] Kaydedildi: {save_path}")

    plt.close()


# ── Eğitim Dinamikleri Analizi ──────────────────────────────────────────

def plot_training_analysis(
    history: Dict,
    model_name: str,
    save_path: str = None
) -> None:
    """
    Detaylı eğitim analizi:
    - Loss/Accuracy eğrileri
    - Overfitting gap analizi (train-val farkı)
    - Learning rate değişimi
    - Generalization gap

    Hoca sorusu: "Modelin ezberlediğini nasıl anlarsın?"
    → Train acc yükselirken val acc düşüyor veya sabit kalıyorsa = overfitting
    → Bu grafikte net görünür
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", lw=2, label="Val Loss")
    ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                    alpha=0.1, color="orange", label="Overfitting Gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Eğrileri ve Overfitting Gap", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy Curves
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", lw=2, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], "r-", lw=2, label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Eğrileri", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 3. Generalization Gap
    ax = axes[1, 0]
    gen_gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    colors_gap = ["#ef4444" if g > 0.1 else "#f59e0b" if g > 0.05 else "#10b981"
                  for g in gen_gap]
    ax.bar(epochs, gen_gap, color=colors_gap, alpha=0.7)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Tehlike (>10%)")
    ax.axhline(y=0.05, color="orange", linestyle="--", alpha=0.5, label="Uyarı (>5%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Acc - Val Acc")
    ax.set_title("Generalization Gap (Düşük = İyi)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], "g-", lw=2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Overfitting durum analizi
    final_gap = gen_gap[-1] if gen_gap else 0
    max_gap = max(gen_gap) if gen_gap else 0
    if max_gap > 0.1:
        status = "UYARI: Overfitting belirtisi (gap > %10)"
        status_color = "red"
    elif max_gap > 0.05:
        status = "DIKKAT: Hafif overfitting egilimi (gap > %5)"
        status_color = "orange"
    else:
        status = "BASARILI: Iyi genelleme (gap < %5)"
        status_color = "green"

    fig.suptitle(
        f"Egitim Dinamikleri Analizi - {model_name}\n{status}",
        fontsize=14, fontweight="bold", y=1.02, color=status_color
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ANALYSIS] Kaydedildi: {save_path}")
        print(f"[ANALYSIS] Durum: {status}")
        print(f"[ANALYSIS] Max generalization gap: {max_gap:.4f}, "
              f"Final gap: {final_gap:.4f}")

    plt.close()


# ── Veri Seti İstatistikleri ────────────────────────────────────────────

def plot_dataset_overview(
    train_labels: List[int],
    val_labels: List[int],
    test_labels: List[int],
    save_path: str = None
) -> None:
    """Veri seti bölümleme istatistikleri ve görselleştirmesi."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sets = {
        "Train": train_labels,
        "Validation": val_labels,
        "Test": test_labels
    }

    colors = ["#3b82f6", "#ef4444"]

    # 1. Sınıf dağılımı (her set)
    ax = axes[0]
    x = np.arange(len(sets))
    width = 0.35
    normal_counts = [sum(1 for l in labels if l == 0) for labels in sets.values()]
    hemorrhage_counts = [sum(1 for l in labels if l == 1) for labels in sets.values()]

    ax.bar(x - width/2, normal_counts, width, label="Normal", color=colors[0])
    ax.bar(x + width/2, hemorrhage_counts, width, label="Hemorrhage", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(sets.keys())
    ax.set_ylabel("Örnek Sayısı")
    ax.set_title("Sınıf Dağılımı (Stratified Split)", fontweight="bold")
    ax.legend()

    # 2. Oran pasta grafiği
    ax = axes[1]
    sizes = [len(l) for l in sets.values()]
    explode = (0.05, 0.05, 0.05)
    ax.pie(sizes, explode=explode, labels=sets.keys(),
           autopct='%1.1f%%', colors=["#3b82f6", "#f59e0b", "#10b981"],
           shadow=True, startangle=90, textprops={'fontsize': 11})
    ax.set_title("Split Oranları (70/15/15)", fontweight="bold")

    # 3. Sınıf oranı doğrulama
    ax = axes[2]
    hemorrhage_ratios = [sum(1 for l in labels if l == 1) / len(labels) * 100
                        for labels in sets.values()]
    bars = ax.bar(list(sets.keys()), hemorrhage_ratios,
                  color=["#3b82f6", "#f59e0b", "#10b981"])
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="Hedef: %50")
    ax.set_ylabel("Hemorrhage Oranı (%)")
    ax.set_title("Sınıf Oranı Korunumu\n(Stratified Split Doğrulama)", fontweight="bold")
    ax.set_ylim([0, 100])
    ax.legend()

    for bar, ratio in zip(bars, hemorrhage_ratios):
        ax.annotate(f'{ratio:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[DATASET] Kaydedildi: {save_path}")

    plt.close()


if __name__ == "__main__":
    print("[VIZ] Bu modül eğitimden sonra çalıştırılmalıdır.")
