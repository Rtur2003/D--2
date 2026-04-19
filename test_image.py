"""
Tek görüntü veya klasör üzerinde 3 modeli test et.

Kullanım:
  python test_image.py                        # interaktif dosya yolu sor
  python test_image.py goruntu.jpg            # tek görüntü
  python test_image.py external_test/         # klasör (normal/ hemorrhage/ alt dizinli)
"""

import sys
import os
import io
import json
import glob
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODELS_DIR, IMG_SIZE
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model

# ── Renkler (terminal) ───────────────────────────────────────────────────
G  = "\033[92m"   # yeşil
R  = "\033[91m"   # kırmızı
Y  = "\033[93m"   # sarı
B  = "\033[94m"   # mavi
W  = "\033[97m"   # beyaz
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"


# ── Model yükleme ────────────────────────────────────────────────────────
def _load_models():
    with open(str(MODELS_DIR / "train_stats.json")) as f:
        stats = json.load(f)

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])

    def _cnx():
        m = get_convnext_model(pretrained=False)
        ck = torch.load(str(MODELS_DIR / "convnext_tiny_best.pth"),
                        map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        return m.eval().to(DEVICE)

    def _cnn(fname):
        p = MODELS_DIR / fname
        if not p.exists():
            return None
        m = get_custom_cnn()
        ck = torch.load(str(p), map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        return m.eval().to(DEVICE)

    models = {
        "ConvNeXt-Tiny     ": _cnx(),
        "Custom CNN Az Veri": _cnn("custom_cnn_best.azveri.pth"),
        "Custom CNN CokVeri": _cnn("custom_cnn_best.cok.veri.pth"),
    }
    # None olanları çıkar
    models = {k: v for k, v in models.items() if v is not None}
    return models, tf


# ── Tek görüntü tahmini ──────────────────────────────────────────────────
@torch.no_grad()
def predict_image(path, models, tf):
    img = Image.open(path).convert("RGB")
    t = tf(img).unsqueeze(0).to(DEVICE)
    preds = {}
    for name, model in models.items():
        p = torch.softmax(model(t), dim=1)[0].cpu().numpy()
        preds[name] = p
    # Ensemble: ConvNeXt + Az Veri (veya mevcut ilk 2)
    keys = list(preds.keys())
    ens = np.mean([preds[k] for k in keys[:2]], axis=0)
    preds["Ensemble          "] = ens
    return preds


def _bar(prob, width=24):
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _conf_color(prob):
    if prob >= 0.85:
        return G
    if prob >= 0.60:
        return Y
    return R


def print_single(path, preds, true_label=None):
    fname = Path(path).name
    print(f"\n{BOLD}{'─'*58}{RST}")
    print(f"  {W}{BOLD}{fname}{RST}")
    if true_label is not None:
        lbl_str = f"{G}Normal{RST}" if true_label == 0 else f"{R}Hemorrhage{RST}"
        print(f"  Gerçek etiket : {lbl_str}")
    print(f"{'─'*58}")

    for model_name, probs in preds.items():
        pred_idx = int(probs.argmax())
        pred_str = "Normal" if pred_idx == 0 else "Hemorrhage"
        conf = float(probs.max())
        c = _conf_color(conf)

        correct_mark = ""
        if true_label is not None:
            correct_mark = f"  {G}✓{RST}" if pred_idx == true_label else f"  {R}✗{RST}"

        bar_n = _bar(probs[0])
        bar_h = _bar(probs[1])

        print(f"\n  {DIM}{model_name}{RST}")
        print(f"  Tahmin : {c}{BOLD}{pred_str:<11}{RST}  güven: {c}{conf:.1%}{RST}{correct_mark}")
        print(f"  Normal     {bar_n} {probs[0]:.1%}")
        print(f"  Hemorrhage {bar_h} {probs[1]:.1%}")

    print(f"{'─'*58}")


# ── Klasör testi ─────────────────────────────────────────────────────────
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect_folder(folder):
    """normal/ ve hemorrhage/ alt klasörlerinden görüntü topla."""
    imgs, labels = [], []
    for sub, lbl in [("normal", 0), ("hemorrhage", 1)]:
        d = Path(folder) / sub
        if d.exists():
            for ext in EXTS:
                for p in sorted(d.glob(ext)):
                    imgs.append(str(p)); labels.append(lbl)
    if not imgs:
        # alt klasör yoksa tüm görüntüleri label=-1 ile al
        for ext in EXTS:
            for p in sorted(Path(folder).glob(ext)):
                imgs.append(str(p)); labels.append(-1)
    return imgs, labels


def print_summary(all_preds, true_labels, model_names):
    n = len(all_preds)
    if n == 0:
        return

    print(f"\n{BOLD}{'═'*58}")
    print(f"  ÖZET — {n} görüntü")
    print(f"{'═'*58}{RST}")

    if true_labels[0] != -1:
        print(f"  {'Model':<22} {'Acc':>6}  {'N-Rec':>7}  {'H-Rec':>7}  {'Sonuç':>8}")
        print(f"  {'─'*54}")

        n_idx = [i for i, l in enumerate(true_labels) if l == 0]
        h_idx = [i for i, l in enumerate(true_labels) if l == 1]

        for mname in model_names:
            preds_list = [r[mname] for r in all_preds]
            correct  = sum(p.argmax() == true_labels[i] for i, p in enumerate(preds_list))
            n_rec = sum(preds_list[i].argmax() == 0 for i in n_idx) / max(len(n_idx), 1)
            h_rec = sum(preds_list[i].argmax() == 1 for i in h_idx) / max(len(h_idx), 1)
            acc = correct / n
            c = G if acc >= 0.85 else (Y if acc >= 0.70 else R)
            print(f"  {DIM}{mname}{RST}  {c}{acc:>5.1%}{RST}  {n_rec:>6.1%}  {h_rec:>6.1%}  {correct:>4}/{n}")

    print(f"\n  Normal: {true_labels.count(0)}  "
          f"Hemorrhage: {true_labels.count(1)}  "
          f"Toplam: {n}")
    print(f"{BOLD}{'═'*58}{RST}\n")


# ── Ana akış ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{B}Head CT Hemorrhage — Model Test{RST}")
    print(f"{DIM}ConvNeXt · Custom CNN Az Veri · Custom CNN Çok Veri · Ensemble{RST}\n")

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input("Dosya veya klasör yolu girin: ").strip().strip('"')

    if not target:
        print("Yol girilmedi, çıkılıyor.")
        return

    print(f"\n{DIM}Modeller yükleniyor...{RST}")
    models, tf = _load_models()
    model_names = list(models.keys()) + ["Ensemble          "]
    print(f"{G}✓ {len(models)} model yüklendi{RST}")

    target_path = Path(target)

    if target_path.is_dir():
        imgs, labels = collect_folder(target_path)
        if not imgs:
            print(f"{R}Klasörde görüntü bulunamadı.{RST}")
            return
        print(f"\n{DIM}Klasör: {target_path}  ({len(imgs)} görüntü){RST}")
        all_preds = []
        for path, lbl in zip(imgs, labels):
            preds = predict_image(path, models, tf)
            print_single(path, preds, lbl if lbl != -1 else None)
            all_preds.append(preds)
        print_summary(all_preds, labels, model_names)

    elif target_path.is_file():
        preds = predict_image(str(target_path), models, tf)
        print_single(str(target_path), preds)

    else:
        print(f"{R}Geçersiz yol: {target}{RST}")


if __name__ == "__main__":
    main()
