# =========================================================================
# EVALUATION (Değerlendirme)
# =========================================================================
# Test seti üzerinde performans metrikleri:
# - Confusion Matrix
# - Accuracy, Precision, Recall, F1-Score
#
# KURAL (Ders Notu Bölüm 7, Adım 5):
# - Test seti sadece FINAL değerlendirme için kullanılır
# - Model seçimi validation ile yapılır, test'e bir kez bakılır
# - Accuracy tek başına yeterli değil → F1, Recall, Precision gerekli
#   (Bölüm 5 - Metrics)
# =========================================================================

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE, DEFAULT_HPARAMS
)
from data_split import get_split_data
from data_preprocessing import (
    HeadCTDataset, get_transforms
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple:
    """Model ile tahmin yap."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
) -> None:
    """Confusion Matrix çiz ve kaydet."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_xlabel("Tahmin (Predicted)", fontsize=13)
    ax.set_ylabel("Gerçek (Actual)", fontsize=13)
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=15, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Confusion Matrix kaydedildi: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict:
    """Performans metriklerini hesapla."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Sınıf bazlı metrikler
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "per_class": {}
    }

    for i, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i])
        }

    return metrics


def print_evaluation_report(metrics: Dict) -> None:
    """Değerlendirme raporunu yazdır."""
    print(f"\n{'='*60}")
    print(f"TEST SONUÇLARI: {metrics['model_name']}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    print(f"\n  Sınıf Bazlı Sonuçlar:")
    print(f"  {'Sınıf':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*45}")
    for name, vals in metrics["per_class"].items():
        print(f"  {name:<15} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1']:>10.4f}")
    print(f"{'='*60}\n")


def evaluate_model(
    model: nn.Module,
    model_name: str,
    test_paths: List[str],
    test_labels: List[int],
    mean: List[float],
    std: List[float]
) -> Dict:
    """Tek bir modeli test seti üzerinde değerlendir."""

    test_transform = get_transforms(mean, std, is_train=False, augment=False)
    test_dataset = HeadCTDataset(test_paths, test_labels, test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=False
    )

    model = model.to(DEVICE)
    y_pred, y_true, y_prob = predict(model, test_loader, DEVICE)

    # Metrikler
    metrics = compute_metrics(y_true, y_pred, model_name)
    print_evaluation_report(metrics)

    # Confusion Matrix - dosya adi icin guvenli isim
    safe_name = model_name.replace("-", "_").replace(" ", "_").lower()
    plot_confusion_matrix(
        y_true, y_pred, model_name,
        save_path=str(RESULTS_DIR / f"{safe_name}_confusion_matrix.png")
    )

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(f"\nDetaylı Classification Report:\n{report}")

    # Metrikleri kaydet
    metrics_path = RESULTS_DIR / f"{safe_name}_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_evaluation() -> None:
    """Her iki model + ensemble için test değerlendirmesi + ileri görselleştirmeler."""
    from visualizations import (
        FeatureExtractor, plot_tsne, plot_roc_curves,
        plot_training_analysis, plot_dataset_overview
    )
    from gradcam import visualize_gradcam_grid
    from ensemble import EnsembleModel, compute_optimal_weights
    from train import create_dataloaders

    # Data split
    train_df, val_df, test_df = get_split_data()
    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["label"].tolist()
    val_paths = val_df["image_path"].tolist()
    val_labels = val_df["label"].tolist()
    test_paths = test_df["image_path"].tolist()
    test_labels = test_df["label"].tolist()

    # Normalizasyon istatistikleri
    stats_path = MODELS_DIR / "train_stats.json"
    with open(str(stats_path), "r") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    print(f"\n[EVAL] Test seti: {len(test_paths)} görüntü")

    # ── Dataset Overview Grafiği ───────────────────────────────────────
    print("\n[EVAL] Veri seti istatistikleri...")
    plot_dataset_overview(
        train_labels, val_labels, test_labels,
        save_path=str(RESULTS_DIR / "dataset_overview.png")
    )

    # ── Eğitim Dinamikleri Analizi ─────────────────────────────────────
    for name in ["convnext_tiny", "custom_cnn"]:
        history_path = RESULTS_DIR / f"{name}_history.json"
        if history_path.exists():
            with open(str(history_path), "r") as f:
                history = json.load(f)
            display = "ConvNeXt-Tiny" if "convnext" in name else "Custom CNN"
            plot_training_analysis(
                history, display,
                save_path=str(RESULTS_DIR / f"{name}_training_analysis.png")
            )

    all_metrics = {}
    test_transform = get_transforms(mean, std, is_train=False, augment=False)

    # ── Model 1: ConvNeXt ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("DEĞERLENDİRME: ConvNeXt-Tiny")
    print("="*70)

    convnext = get_convnext_model(pretrained=False)
    checkpoint = torch.load(str(MODELS_DIR / "convnext_tiny_best.pth"), map_location=DEVICE, weights_only=False)
    convnext.load_state_dict(checkpoint["model_state_dict"])

    metrics1 = evaluate_model(
        convnext, "ConvNeXt-Tiny", test_paths, test_labels, mean, std
    )
    all_metrics["convnext"] = metrics1

    # ── Model 2: Custom CNN ────────────────────────────────────────────
    print("\n" + "="*70)
    print("DEĞERLENDİRME: Custom CNN")
    print("="*70)

    custom_cnn = get_custom_cnn()
    checkpoint = torch.load(str(MODELS_DIR / "custom_cnn_best.pth"), map_location=DEVICE, weights_only=False)
    custom_cnn.load_state_dict(checkpoint["model_state_dict"])

    metrics2 = evaluate_model(
        custom_cnn, "Custom CNN", test_paths, test_labels, mean, std
    )
    all_metrics["custom_cnn"] = metrics2

    # ── Ensemble ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DEĞERLENDİRME: Ensemble (ConvNeXt + Custom CNN)")
    print("="*70)

    # Validation üzerinden optimal ağırlık bul
    val_loader_for_ensemble = DataLoader(
        HeadCTDataset(val_paths, val_labels,
                      get_transforms(mean, std, is_train=False)),
        batch_size=16, shuffle=False, num_workers=0
    )
    w1, w2 = compute_optimal_weights(convnext, custom_cnn, val_loader_for_ensemble)
    ensemble = EnsembleModel(convnext, custom_cnn, weight1=w1, weight2=w2)

    test_loader = DataLoader(
        HeadCTDataset(test_paths, test_labels,
                      get_transforms(mean, std, is_train=False)),
        batch_size=16, shuffle=False, num_workers=0
    )
    ens_preds, ens_labels, ens_probs = ensemble.predict_loader(test_loader)

    ens_metrics = compute_metrics(ens_labels, ens_preds, "Ensemble")
    print_evaluation_report(ens_metrics)
    plot_confusion_matrix(
        ens_labels, ens_preds, "Ensemble",
        save_path=str(RESULTS_DIR / "ensemble_confusion_matrix.png")
    )
    all_metrics["ensemble"] = ens_metrics

    # ── Karşılaştırma Tablosu ──────────────────────────────────────────
    print("\n" + "="*70)
    print("MODEL KARŞILAŞTIRMASI (3 Model)")
    print("="*70)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*65}")
    for key, m in all_metrics.items():
        print(f"  {m['model_name']:<25} {m['accuracy']:>10.4f} "
              f"{m['precision_weighted']:>10.4f} {m['recall_weighted']:>10.4f} "
              f"{m['f1_weighted']:>10.4f}")
    print(f"{'='*70}")

    plot_comparison(all_metrics)

    # ── ROC-AUC Eğrileri ───────────────────────────────────────────────
    print("\n[EVAL] ROC-AUC eğrileri oluşturuluyor...")
    roc_data = {}

    # Her model için hemorrhage (sınıf 1) olasılığını al
    for name, model in [("ConvNeXt-Tiny", convnext), ("Custom CNN", custom_cnn)]:
        y_pred, y_true, y_prob = predict(model, test_loader, DEVICE)
        roc_data[name] = (y_true, y_prob[:, 1])

    roc_data["Ensemble"] = (ens_labels, ens_probs[:, 1])

    plot_roc_curves(
        roc_data,
        save_path=str(RESULTS_DIR / "roc_auc_curves.png")
    )

    # ── t-SNE Feature Visualization ────────────────────────────────────
    print("\n[EVAL] t-SNE feature visualization...")
    for name, model in [("ConvNeXt-Tiny", convnext), ("Custom CNN", custom_cnn)]:
        try:
            extractor = FeatureExtractor(model, name)
            features, labels = extractor.extract(test_loader)
            safe_name = name.replace("-", "_").replace(" ", "_").lower()
            plot_tsne(
                features, labels, name,
                save_path=str(RESULTS_DIR / f"{safe_name}_tsne.png")
            )
        except Exception as e:
            print(f"[t-SNE] {name} için hata: {e}")

    # ── Grad-CAM ───────────────────────────────────────────────────────
    print("\n[EVAL] Grad-CAM görselleştirmesi...")
    for name, model in [("ConvNeXt-Tiny", convnext), ("Custom CNN", custom_cnn)]:
        try:
            safe_name = name.replace("-", "_").replace(" ", "_").lower()
            visualize_gradcam_grid(
                model, name, test_paths, test_labels, test_transform,
                save_path=str(RESULTS_DIR / f"{safe_name}_gradcam.png"),
                num_samples=8
            )
        except Exception as e:
            print(f"[GRAD-CAM] {name} için hata: {e}")

    print("\n" + "="*70)
    print("TÜM DEĞERLENDİRMELER TAMAMLANDI!")
    print("="*70)


def plot_comparison(all_metrics: Dict) -> None:
    """Tum modellerin metriklerini karsilastiran bar chart."""
    models = [m["model_name"] for m in all_metrics.values()]
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"]

    values = []
    for m in all_metrics.values():
        values.append([
            m["accuracy"], m["precision_weighted"],
            m["recall_weighted"], m["f1_weighted"]
        ])

    n_models = len(models)
    x = np.arange(len(metrics_names))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    all_bars = []
    for i in range(n_models):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values[i], width,
                      label=models[i], color=colors[i % len(colors)])
        all_bars.append(bars)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Karsilastirmasi - Test Sonuclari",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", alpha=0.3)

    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = str(RESULTS_DIR / "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Karsilastirma grafigi kaydedildi: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_evaluation()
