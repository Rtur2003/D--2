"""
v2 split loader: hasta-bazli train/val/test.csv'leri dogrudan okur.

Stratified + hasta-bazli bolme zaten v2/scripts/split.py ile yapildi
(slice leakage'i onlemek icin hasta seviyesinde). Bu modul sadece
onceden uretilmis CSV'leri DataFrame'e yukler.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> pd.DataFrame:
    path = V2_ROOT / f"{name}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"path": "image_path"})
    return df[["image_path", "label"]]


def get_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = _load("train")
    va = _load("val")
    te = _load("test")
    print_split_summary(tr, va, te)
    return tr, va, te


def print_split_summary(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n" + "=" * 60)
    print("DATA SPLIT (v2, hasta-bazli)")
    print("=" * 60)
    print(f"{'Set':<12} {'Toplam':>8} {'Normal':>8} {'Hemorrhage':>12} {'Oran':>8}")
    print("-" * 60)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        r = len(df) / total * 100
        print(f"{name:<12} {len(df):>8} {n0:>8} {n1:>12} {r:>7.1f}%")
    print("-" * 60)
    print(f"{'TOPLAM':<12} {total:>8}\n")


if __name__ == "__main__":
    get_split_data()
