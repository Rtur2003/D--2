# =========================================================================
# HYPERPARAMETER TUNING (Hiperparametre Optimizasyonu)
# =========================================================================
# Grid Search ile en iyi hiperparametrelerin bulunması.
#
# KURALLAR (HPO Ders Notundan):
# - Grid Search: Tüm kombinasyonları dener, küçük uzay için uygun
# - Model seçimi VALIDATION seti üzerinden yapılır
# - Test seti ASLA tuning'de kullanılmaz
# - En iyi validation skorunun tek başına yeterli olmadığı unutulmamalı;
#   seed duyarlılığı ve validation'a aşırı uyum sorgulanmalı
# =========================================================================

import json
import itertools
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, HPARAM_GRID,
    DEFAULT_HPARAMS, RANDOM_SEED
)
from data_split import get_split_data
from data_preprocessing import (
    HeadCTDataset, compute_train_statistics, get_transforms
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model
from train import train_one_epoch, validate, set_seed, create_dataloaders

import matplotlib.pyplot as plt


def grid_search(
    model_fn,
    model_name: str,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    mean: List[float],
    std: List[float],
    param_grid: Dict = None,
    quick_epochs: int = 15
) -> Tuple[Dict, List[Dict]]:
    """
    Grid Search - Tüm hiperparametre kombinasyonlarını dene.

    Her kombinasyon için:
    1. Model oluştur
    2. quick_epochs kadar eğit
    3. Validation loss/accuracy kaydet
    4. En iyi kombinasyonu seç

    Not: Quick search (15 epoch) → en iyi parametrelerle tam eğitim yapılır
    """
    if param_grid is None:
        param_grid = HPARAM_GRID

    # Tüm kombinasyonları oluştur
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {model_name}")
    print(f"{'='*60}")
    print(f"Parametre uzayı:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    print(f"Toplam kombinasyon: {len(combinations)}")
    print(f"Her biri {quick_epochs} epoch eğitilecek")
    print(f"{'='*60}\n")

    results = []
    best_val_loss = float("inf")
    best_params = None

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        hparams = {**DEFAULT_HPARAMS, **params}

        print(f"\n[{idx}/{len(combinations)}] Deneniyor: {params}")

        set_seed(RANDOM_SEED)

        # DataLoader
        train_loader, val_loader = create_dataloaders(
            train_paths, train_labels,
            val_paths, val_labels,
            mean, std,
            batch_size=hparams["batch_size"],
            augment=True
        )

        # Model
        model = model_fn().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # Quick eğitim
        best_epoch_val_loss = float("inf")
        best_epoch_val_acc = 0.0

        for epoch in range(1, quick_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, DEVICE
            )

            if val_loss < best_epoch_val_loss:
                best_epoch_val_loss = val_loss
                best_epoch_val_acc = val_acc

        result = {
            **params,
            "best_val_loss": float(best_epoch_val_loss),
            "best_val_acc": float(best_epoch_val_acc)
        }
        results.append(result)

        print(f"  -> Val Loss: {best_epoch_val_loss:.4f}, "
              f"Val Acc: {best_epoch_val_acc:.4f}")

        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_params = hparams
            print(f"  [BEST] Yeni en iyi!")

    # Sonuçları sırala
    results.sort(key=lambda x: x["best_val_loss"])

    print(f"\n{'='*60}")
    print(f"GRID SEARCH SONUÇLARI: {model_name}")
    print(f"{'='*60}")
    print(f"{'LR':>10} {'Batch':>6} {'W.Decay':>10} {'Val Loss':>10} {'Val Acc':>10}")
    print(f"{'-'*50}")
    for r in results:
        print(f"{r['learning_rate']:>10.1e} {r['batch_size']:>6} "
              f"{r['weight_decay']:>10.1e} {r['best_val_loss']:>10.4f} "
              f"{r['best_val_acc']:>10.4f}")

    print(f"\n[BEST] En iyi parametreler:")
    print(f"  Learning Rate: {best_params['learning_rate']}")
    print(f"  Batch Size: {best_params['batch_size']}")
    print(f"  Weight Decay: {best_params['weight_decay']}")
    print(f"{'='*60}")

    return best_params, results


def plot_grid_search_results(
    results: List[Dict],
    model_name: str,
    save_path: str = None
) -> None:
    """Grid Search sonuçlarını görselleştir."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Grid Search Sonuçları - {model_name}",
                 fontsize=14, fontweight="bold")

    # Learning Rate vs Val Loss
    lrs = [r["learning_rate"] for r in results]
    losses = [r["best_val_loss"] for r in results]
    axes[0].scatter(lrs, losses, c=losses, cmap="RdYlGn_r", s=80, edgecolors="black")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate", fontsize=11)
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    axes[0].set_title("LR vs Val Loss", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Batch Size vs Val Acc
    batches = sorted(set(r["batch_size"] for r in results))
    batch_accs = {}
    for b in batches:
        accs = [r["best_val_acc"] for r in results if r["batch_size"] == b]
        batch_accs[b] = accs

    positions = list(range(len(batches)))
    bp = axes[1].boxplot(
        [batch_accs[b] for b in batches],
        positions=positions, widths=0.6
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(b) for b in batches])
    axes[1].set_xlabel("Batch Size", fontsize=11)
    axes[1].set_ylabel("Validation Accuracy", fontsize=11)
    axes[1].set_title("Batch Size vs Val Acc", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Top 10 combinations
    top10 = results[:10]
    labels = [f"LR={r['learning_rate']:.0e}\nBS={r['batch_size']}" for r in top10]
    accs = [r["best_val_acc"] for r in top10]
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(top10)))
    axes[2].barh(range(len(top10)), accs, color=colors)
    axes[2].set_yticks(range(len(top10)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel("Validation Accuracy", fontsize=11)
    axes[2].set_title("Top 10 Kombinasyon", fontsize=12)
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Grid Search sonuçları kaydedildi: {save_path}")
    else:
        plt.show()

    plt.close()


def run_hyperparameter_tuning() -> Dict:
    """Ana HPO pipeline'ı."""
    set_seed()

    # Data split
    train_df, val_df, test_df = get_split_data()
    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["label"].tolist()
    val_paths = val_df["image_path"].tolist()
    val_labels = val_df["label"].tolist()

    # Normalizasyon
    mean, std = compute_train_statistics(train_paths)

    # ── ConvNeXt Grid Search ───────────────────────────────────────────
    convnext_best, convnext_results = grid_search(
        model_fn=lambda: get_convnext_model(pretrained=True),
        model_name="ConvNeXt-Tiny",
        train_paths=train_paths, train_labels=train_labels,
        val_paths=val_paths, val_labels=val_labels,
        mean=mean, std=std,
        quick_epochs=5
    )

    plot_grid_search_results(
        convnext_results, "ConvNeXt-Tiny",
        save_path=str(RESULTS_DIR / "convnext_grid_search.png")
    )

    # ── Custom CNN Grid Search ─────────────────────────────────────────
    custom_best, custom_results = grid_search(
        model_fn=get_custom_cnn,
        model_name="Custom CNN",
        train_paths=train_paths, train_labels=train_labels,
        val_paths=val_paths, val_labels=val_labels,
        mean=mean, std=std,
        quick_epochs=5
    )

    plot_grid_search_results(
        custom_results, "Custom CNN",
        save_path=str(RESULTS_DIR / "custom_cnn_grid_search.png")
    )

    # Sonuçları kaydet
    tuning_results = {
        "convnext": {
            "best_params": convnext_best,
            "all_results": convnext_results
        },
        "custom_cnn": {
            "best_params": custom_best,
            "all_results": custom_results
        }
    }

    results_path = RESULTS_DIR / "hyperparameter_tuning_results.json"
    with open(str(results_path), "w") as f:
        json.dump(tuning_results, f, indent=2, default=str)

    print(f"\n[SAVE] HPO sonuçları kaydedildi: {results_path}")
    return tuning_results


if __name__ == "__main__":
    run_hyperparameter_tuning()
