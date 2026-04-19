"""
Rapor icin akis semasi ve mimari diyagramlari uretir.

Ciktilar (results/ altina):
  - flow_chart.png            : Tum pipeline akis semasi
  - architecture_convnext.png : ConvNeXt-Tiny katman ozeti
  - architecture_custom_cnn.png : Custom CNN mimari diyagrami
  - decision_pipeline.png     : Tek goruntu icin inference akisi
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import RESULTS_DIR


# ── Renk paleti (sade, klinik) ──────────────────────────────────────────

PALETTE = {
    "data": "#0f766e",
    "preprocess": "#0891b2",
    "split": "#7c3aed",
    "augment": "#0d9488",
    "model": "#1e40af",
    "tune": "#b45309",
    "train": "#1e3a5f",
    "eval": "#be123c",
    "ui": "#475569",
    "bg": "#f8fafc",
    "text": "#0f172a",
    "muted": "#64748b",
    "edge": "#cbd5e1",
}


def _box(ax, x, y, w, h, label, color, fontsize=10, text_color="white", style="round,pad=0.3"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        linewidth=1.2,
        facecolor=color,
        edgecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=text_color,
        weight="600", family="DejaVu Sans",
    )


def _arrow(ax, x1, y1, x2, y2, color="#94a3b8", style="-|>", lw=1.4):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=14,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)


# ── 1. Tum pipeline akis semasi ─────────────────────────────────────────

def make_flow_chart(save_path: Path):
    fig, ax = plt.subplots(figsize=(13, 14))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # 1. Veri seti
    _box(ax, 1, 12.6, 12, 0.8,
         "Head CT Hemorrhage  ·  200 PNG  ·  100 Normal + 100 Hemorrhage",
         PALETTE["data"], fontsize=11)

    # 2. Preprocess
    _box(ax, 1, 11.3, 12, 0.8,
         "On Isleme  ·  Resize 224x224  ·  RGB  ·  Train mean/std normalize",
         PALETTE["preprocess"], fontsize=11)

    # 3. Split
    _box(ax, 1, 10.0, 12, 0.8,
         "Stratified Split 70 / 15 / 15  ·  139 / 31 / 30",
         PALETTE["split"], fontsize=11)

    # 4. Train side
    _box(ax, 0.3, 8.6, 6.2, 0.8,
         "Train  ->  Augmentation (CLAHE · Affine · Flip · Noise)",
         PALETTE["augment"], fontsize=10)

    # 4b. Val/Test side
    _box(ax, 7.5, 8.6, 6.2, 0.8,
         "Val + Test  ->  augmentation YOK (ham dagilim)",
         PALETTE["muted"], fontsize=10)

    # 5. HPO
    _box(ax, 0.3, 7.3, 6.2, 0.8,
         "Grid Search 12 kombinasyon  +  5-fold CV",
         PALETTE["tune"], fontsize=10)

    # 6. Model 1
    _box(ax, 0.3, 5.7, 6.2, 1.2,
         "Model 1: ConvNeXt-Tiny\n"
         "ImageNet pretrained  ·  28M param  ·  Progressive Unfreeze",
         PALETTE["model"], fontsize=9.5)

    # 6b. Model 2
    _box(ax, 7.5, 5.7, 6.2, 1.2,
         "Model 2: Custom CNN\n"
         "Stem  ·  MultiScale  ·  3x ResidualSE  ·  1.3M param",
         PALETTE["model"], fontsize=9.5)

    # 7. Train techniques
    _box(ax, 1, 4.3, 12, 0.9,
         "Egitim  ·  AdamW  ·  Mixup  ·  Label Smoothing  ·  "
         "Cosine Annealing  ·  Grad Clip  ·  Early Stop",
         PALETTE["train"], fontsize=10)

    # 8. Eval
    _box(ax, 1, 1.4, 12, 1.2,
         "Test (30 goruntu)  ·  Confusion Matrix  ·  "
         "Precision / Recall / F1\n"
         "ROC-AUC  ·  PR-AUC  ·  t-SNE  ·  Grad-CAM  ·  Threshold Tuning",
         PALETTE["eval"], fontsize=10)

    # 10. UI
    _box(ax, 1, 0.1, 12, 0.8,
         "Gradio Karar Destek Arayuzu  ·  Tek dosya tahmini + olasilik",
         PALETTE["ui"], fontsize=11)

    # Arrows (vertical chain)
    chain = [
        (7, 12.6, 7, 12.1),   # data -> preprocess
        (7, 11.3, 7, 10.8),   # preprocess -> split
    ]
    for x1, y1, x2, y2 in chain:
        _arrow(ax, x1, y1, x2, y2)

    # Split branches: split bottom -> train and val/test
    _arrow(ax, 5, 10.0, 3.4, 9.4)
    _arrow(ax, 9, 10.0, 10.6, 9.4)

    # Train side flow
    _arrow(ax, 3.4, 8.6, 3.4, 8.1)   # aug -> hpo
    _arrow(ax, 3.4, 7.3, 3.4, 6.9)   # hpo -> model 1

    # HPO -> model 2 (cross arrow)
    _arrow(ax, 4.5, 7.3, 9, 6.9, color="#cbd5e1", lw=1.0)

    # Val/Test side joins eval directly (long down arrow)
    _arrow(ax, 10.6, 8.6, 10.6, 2.6, color="#cbd5e1", lw=1.0)

    # Models -> training (converge to center)
    _arrow(ax, 3.4, 5.7, 5.5, 5.2)
    _arrow(ax, 10.6, 5.7, 8.5, 5.2)

    # training -> eval -> ui
    _arrow(ax, 7, 4.3, 7, 2.6)
    _arrow(ax, 7, 1.4, 7, 0.9)

    # Title
    fig.suptitle(
        "Head CT Hemorrhage Classification  —  Pipeline",
        fontsize=14, weight="700", color=PALETTE["text"], y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "BM 480 Derin Ogrenme · Proje 2 · 200-goruntu kucuk veri rejimi",
        ha="center", fontsize=9.5, color=PALETTE["muted"], style="italic",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[DIAG] {save_path}")


# ── 2. ConvNeXt mimari diyagrami ────────────────────────────────────────

def make_convnext_arch(save_path: Path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    blocks = [
        ("Input\n3 x 224 x 224", "#475569"),
        ("Stem\n4x4 Conv s=4", "#0d9488"),
        ("Stage 1\n3 blocks · 96", "#1e40af"),
        ("Stage 2\n3 blocks · 192", "#1e40af"),
        ("Stage 3\n9 blocks · 384", "#1e40af"),
        ("Stage 4\n3 blocks · 768", "#1e40af"),
        ("GAP + LN", "#7c3aed"),
        ("FC 768 -> 2\n(replaced)", "#be123c"),
    ]
    n = len(blocks)
    w = 1.4
    gap = 0.2
    total = n * w + (n - 1) * gap
    x0 = (13 - total) / 2
    y = 2.2
    h = 1.6

    for i, (label, color) in enumerate(blocks):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, color, fontsize=8.5)
        if i < n - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    ax.text(6.5, 4.3,
            "ConvNeXt-Tiny  ·  ImageNet pretrained  ·  ~28M parametre",
            ha="center", fontsize=12, weight="600", color=PALETTE["text"])

    # Strategy footer
    ax.text(
        6.5, 0.9,
        "Transfer Learning Stratejisi:  Faz 1 → backbone donduruldu, sadece head (5 epoch).  "
        "Faz 2 → tamami fine-tune (lr=2e-5, cosine annealing, mixup).",
        ha="center", fontsize=9, color=PALETTE["muted"], style="italic",
    )
    ax.text(
        6.5, 0.4,
        "Karar gerekcesi:  200-goruntu rejimde sifirdan egitim overfitting riskidir → ImageNet ozellikleri buyuk avantaj.",
        ha="center", fontsize=9, color="#0f766e",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[DIAG] {save_path}")


# ── 3. Custom CNN mimari diyagrami ──────────────────────────────────────

def make_custom_cnn_arch(save_path: Path):
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    blocks = [
        ("Input\n3 x 224 x 224", "#475569", 1.4),
        ("Stem\n7x7 Conv s=2\n+ MaxPool\n-> 32x56x56", "#0d9488", 1.6),
        ("MultiScale\n1x1 + 3x3 + 5x5\nfeature fusion\n-> 48x56x56", "#7c3aed", 1.7),
        ("ResSE 1\n48 -> 64\nstride 2\n-> 64x28x28", "#1e40af", 1.55),
        ("ResSE 2\n64 -> 128\nstride 2\n-> 128x14x14", "#1e40af", 1.6),
        ("ResSE 3\n128 -> 256\nstride 2\n-> 256x7x7", "#1e40af", 1.55),
        ("GAP\n256 x 1 x 1", "#7c3aed", 1.3),
        ("FC 256->128\nLayerNorm\nReLU\nDropout 0.4", "#b45309", 1.6),
        ("FC 128 -> 2", "#be123c", 1.4),
    ]

    gap = 0.18
    total = sum(b[2] for b in blocks) + gap * (len(blocks) - 1)
    x = (16 - total) / 2
    y = 2.6
    h = 2.0

    for i, (label, color, w) in enumerate(blocks):
        _box(ax, x, y, w, h, label, color, fontsize=8)
        if i < len(blocks) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap

    ax.text(8, 5.2,
            "Custom CNN  ·  Residual + SE Attention + Multi-Scale  "
            "·  ~1.29M parametre",
            ha="center", fontsize=12, weight="600", color=PALETTE["text"])

    # Design notes
    notes = [
        ("Tasarim secimi", "200 goruntu icin 28M param cok fazla -> kompakt mimari (1.3M) overfitting'i sinirlar.", "#0f766e"),
        ("MultiScale",     "1x1 + 3x3 + 5x5 paralel kollar -> kucuk subdural ve buyuk intraserebral kanamayi ayni anda yakalar.", "#7c3aed"),
        ("SE Attention",   "Squeeze-Excitation kanal bazli 're-weight' yapar -> kanama-iliskili feature map'leri vurgular.", "#1e40af"),
        ("Residual",       "Kaiming init + residual + BatchNorm -> kucuk veride bile stabil gradient akisi.", "#b45309"),
    ]
    for i, (head, body, color) in enumerate(notes):
        ax.text(0.3, 1.7 - i * 0.36, f"\u2022  {head}:",
                fontsize=8.5, weight="600", color=color)
        ax.text(2.4, 1.7 - i * 0.36, body, fontsize=8.5,
                color=PALETTE["text"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[DIAG] {save_path}")


# ── 4. Inference (single image) pipeline ────────────────────────────────

def make_decision_pipeline(save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    steps = [
        ("CT Goruntu", "#475569"),
        ("Resize 224\n+ Normalize", "#0891b2"),
        ("ConvNeXt-Tiny\nsoftmax", "#1e40af"),
        ("Custom CNN\nsoftmax", "#1e40af"),
        ("Grad-CAM\n(aciklanabilirlik)", "#7c3aed"),
        ("Karar\nP(Hem) >= esik", "#be123c"),
    ]

    n = len(steps)
    w = 1.5
    gap = 0.18
    total = n * w + (n - 1) * gap
    x0 = (12 - total) / 2
    y = 1.7
    h = 1.5

    # ConvNeXt and Custom CNN should be parallel — special layout
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, color, fontsize=8.5)
        if i < n - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    ax.text(6, 3.7, "Inference Pipeline (Tek Goruntu)",
            ha="center", fontsize=12, weight="600", color=PALETTE["text"])

    ax.text(
        6, 0.7,
        "Test-Time Augmentation: 4-view ortalama (orijinal + h-flip + v-flip + 180°). "
        "Esik 0.40-0.50 araliginda klinik recall optimizasyonu mumkun.",
        ha="center", fontsize=9, color=PALETTE["muted"], style="italic", wrap=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[DIAG] {save_path}")


# ── Entry ───────────────────────────────────────────────────────────────

def make_all():
    out = Path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    make_flow_chart(out / "flow_chart.png")
    make_convnext_arch(out / "architecture_convnext.png")
    make_custom_cnn_arch(out / "architecture_custom_cnn.png")
    make_decision_pipeline(out / "decision_pipeline.png")


if __name__ == "__main__":
    make_all()
