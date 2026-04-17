"""
Her goruntude beyin dokusu orani hesapla + dagilim raporu.

Beyin dokusu CT'de grayscale ~30-200 aralig'inda; <30 arka plan,
>200 kafatasi/kemik. Eger tissue orani %5'in altindaysa dilim
buyuk olasilikla sadece kemik veya bos (beyin yok) -> egitime uygun degil.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels.csv"
OUT = ROOT / "brain_content_audit.csv"


def tissue_ratio(path: str) -> float:
    img = np.array(Image.open(path).convert("L"))
    # beyin dokusu grayscale ~30-200
    mask = (img > 30) & (img < 200)
    return float(mask.mean())


def main():
    with open(LABELS, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "patient_id", "tissue_ratio"])
        w.writeheader()
        ratios_by_source = defaultdict(list)
        for i, r in enumerate(rows):
            ratio = tissue_ratio(r["path"])
            w.writerow({**r, "tissue_ratio": f"{ratio:.4f}"})
            ratios_by_source[r["source"]].append(ratio)
            if (i + 1) % 1000 == 0:
                print(f"  ... {i+1}/{len(rows)}")

    print("\nDAGILIM (tissue_ratio):")
    for src, vals in ratios_by_source.items():
        arr = np.array(vals)
        below = [0.01, 0.05, 0.10, 0.20]
        print(f"\n{src}  (n={len(arr)})")
        print(f"  min={arr.min():.3f}  median={np.median(arr):.3f}  max={arr.max():.3f}")
        for t in below:
            pct = (arr < t).mean() * 100
            print(f"  <{t:.2f}: {(arr<t).sum():5d} ({pct:5.1f}%)")

    print(f"\nCIKTI: {OUT}")


if __name__ == "__main__":
    main()
