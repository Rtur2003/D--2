"""
Karar esigi (decision threshold) analizi.

Medikal AI'da varsayilan 0.5 esigi optimum degildir:
  - False Negative (kacirilan kanama) cok pahalidir → recall onemli.
  - False Positive sadece ek inceleme demektir → daha az kritik.

Bu script test seti uzerinde 0.05 → 0.95 araliginda esigi tarar ve:
  - precision, recall, F1, balanced accuracy egrilerini cizer
  - Youden's J ile optimum esigi bulur
  - Recall >= 0.95 saglayan minimum esigi raporlar
  - results/threshold_analysis.png ve results/threshold_analysis.json yazar
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from torch.utils.data import DataLoader

from config import DEVICE, MODELS_DIR, RESULTS_DIR
from data_preprocessing import HeadCTDataset, get_transforms
from data_split import get_split_data
from pretrained_model import get_convnext_model


def _load_model():
    with open(str(MODELS_DIR / "train_stats.json"), "r") as f:
        stats = json.load(f)
    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(
        str(MODELS_DIR / "convnext_tiny_best.pth"),
        map_location=DEVICE, weights_only=False
    )
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)
    return convnext, stats


@torch.no_grad()
def _get_probs(test_loader, model):
    p_all, y_all = [], []
    for x, y in test_loader:
        p = torch.nn.functional.softmax(model(x.to(DEVICE)), dim=1)
        p_all.append(p.cpu().numpy())
        y_all.append(y.numpy())
    return np.vstack(p_all), np.concatenate(y_all)


def run_threshold_analysis():
    convnext, stats = _load_model()
    _, _, test_df = get_split_data()
    test_paths = test_df["image_path"].tolist()
    test_labels = test_df["label"].tolist()

    tf = get_transforms(stats["mean"], stats["std"], is_train=False)
    loader = DataLoader(
        HeadCTDataset(test_paths, test_labels, tf),
        batch_size=8, shuffle=False, num_workers=0,
    )

    probs, y_true = _get_probs(loader, convnext)
    p_hem = probs[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (p_hem >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    # Optimum esikler
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_youden = max(rows, key=lambda r: r["recall"] + (1 - r["fp"] / max(1, r["fp"] + r["tn"])) - 1)
    high_recall = next((r for r in rows if r["recall"] >= 0.95), None)
    if high_recall is None:
        high_recall = max(rows, key=lambda r: r["recall"])

    summary = {
        "rows": rows,
        "best_f1_threshold": best_f1,
        "best_youden_threshold": best_youden,
        "min_threshold_recall_0.95": high_recall,
    }
    out_json = RESULTS_DIR / "threshold_analysis.json"
    with open(str(out_json), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[THRESHOLD] {out_json}")

    _plot(rows, summary, RESULTS_DIR / "threshold_analysis.png")
    default = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-3)
    print("\n[THRESHOLD] Onemli esikler:")
    print(f"  Default 0.50  -> precision={default['precision']:.3f}  recall={default['recall']:.3f}  F1={default['f1']:.3f}")
    print(f"  Best F1 {best_f1['threshold']:.2f} -> F1={best_f1['f1']:.3f}  recall={best_f1['recall']:.3f}  precision={best_f1['precision']:.3f}")
    print(f"  Recall>=0.95   -> threshold={high_recall['threshold']:.2f}  precision={high_recall['precision']:.3f}  recall={high_recall['recall']:.3f}")

    return summary


def _plot(rows, summary, save_path):
    t = [r["threshold"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    bal = [r["balanced_acc"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t, prec, "-o", color="#1e40af", label="Precision", lw=2, markersize=4)
    ax.plot(t, rec, "-o", color="#be123c", label="Recall (Sensitivity)", lw=2, markersize=4)
    ax.plot(t, f1, "-o", color="#0d9488", label="F1", lw=2, markersize=4)
    ax.plot(t, bal, "-o", color="#7c3aed", label="Balanced Acc", lw=2, markersize=4)

    ax.axvline(0.5, color="#94a3b8", linestyle=":", lw=1, label="Default 0.50")
    ax.axvline(summary["best_f1_threshold"]["threshold"], color="#0d9488",
               linestyle="--", lw=1, label=f"Best F1 ({summary['best_f1_threshold']['threshold']:.2f})")
    ax.axvline(summary["min_threshold_recall_0.95"]["threshold"], color="#be123c",
               linestyle="--", lw=1, label=f"Recall ≥ 0.95 ({summary['min_threshold_recall_0.95']['threshold']:.2f})")

    ax.set_xlabel("Karar esigi  P(Hemorrhage)", fontsize=11)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title("Threshold tarama  ·  Test seti", fontsize=12, fontweight="600")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)

    # Sag: confusion matrix degisimi (TP, FN, FP, TN)
    ax = axes[1]
    tp = [r["tp"] for r in rows]
    fn = [r["fn"] for r in rows]
    fp = [r["fp"] for r in rows]
    tn = [r["tn"] for r in rows]

    ax.fill_between(t, 0, tp, color="#0d9488", alpha=0.6, label="TP (dogru kanama)")
    ax.fill_between(t, tp, [a + b for a, b in zip(tp, fn)], color="#be123c",
                    alpha=0.6, label="FN (kacirilan kanama)")
    ax.fill_between(t, [a + b for a, b in zip(tp, fn)],
                    [a + b + c for a, b, c in zip(tp, fn, fp)],
                    color="#f59e0b", alpha=0.5, label="FP (yanlis alarm)")

    ax.set_xlabel("Karar esigi  P(Hemorrhage)", fontsize=11)
    ax.set_ylabel("Test seti orneği sayısı", fontsize=11)
    ax.set_title("Confusion bilesenleri  ·  Esige duyarlılık", fontsize=12, fontweight="600")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Decision Threshold Analizi  —  Klinik trade-off (recall vs precision)",
        fontsize=13, fontweight="700", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[THRESHOLD] {save_path}")


if __name__ == "__main__":
    run_threshold_analysis()
