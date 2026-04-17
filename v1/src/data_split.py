# =========================================================================
# DATA SPLIT (Veri Bölümleme)
# =========================================================================
# Stratified Train/Validation/Test split
#
# KURALLAR (Ders Notlarından):
# 1. Küçük veri (200 örnek) → Stratified split zorunlu
# 2. 70/15/15 oranı küçük-dengeli veri için uygun
# 3. Önce test setini ayır, DOKUNMA
# 4. Tüm preprocessing train-merkezli pipeline içinde olmalı
# 5. Sınıf oranları her sette korunmalı
# 6. random_state ile tekrarlanabilirlik sağla
# =========================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, CLASS_NAMES
from data_preprocessing import load_labels


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified Train/Validation/Test split.

    Strateji: Ders Notu Bölüm 2 - Küçük Veri Seti
    - Sınıf oranları korunur (stratify=y)
    - 70/15/15 split (küçük-dengeli veri için önerilen)
    - İlk olarak test seti ayrılır (Bölüm 7, Adım 3)
    - Sonra kalan veri train/val olarak bölünür

    Python Kodu: Ders Notu 2 - Bölüm 2 (Stratified Split)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Oranlar toplamı 1.0 olmalı: {train_ratio + val_ratio + test_ratio}"

    X = df["image_path"].to_numpy()
    y = df["label"].to_numpy()

    # Adım 1: Test setini ayır (önce test → dokunulmaz)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )

    # Adım 2: Kalan veriyi Train/Validation olarak böl
    val_relative = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=random_state
    )

    # DataFrame'lere dönüştür
    train_df = pd.DataFrame({"image_path": X_train, "label": y_train})
    val_df = pd.DataFrame({"image_path": X_val, "label": y_val})
    test_df = pd.DataFrame({"image_path": X_test, "label": y_test})

    return train_df, val_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """Split özetini yazdır."""
    total = len(train_df) + len(val_df) + len(test_df)

    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY (Veri Bölümleme Özeti)")
    print("=" * 60)
    print(f"{'Set':<12} {'Toplam':>8} {'Normal':>8} {'Hemorrhage':>12} {'Oran':>8}")
    print("-" * 60)

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        n_normal = (df["label"] == 0).sum()
        n_hemorrhage = (df["label"] == 1).sum()
        ratio = len(df) / total * 100
        print(f"{name:<12} {len(df):>8} {n_normal:>8} {n_hemorrhage:>12} {ratio:>7.1f}%")

    print("-" * 60)
    print(f"{'TOPLAM':<12} {total:>8}")
    print("=" * 60)

    # Sınıf oranlarını doğrula
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        hemorrhage_ratio = df["label"].mean()
        print(f"[CHECK] {name} hemorrhage oranı: {hemorrhage_ratio:.2%} "
              f"(hedef: ~50.00%)")

    print()


def get_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ana split fonksiyonu - yükle ve böl."""
    df = load_labels()
    train_df, val_df, test_df = stratified_split(df)
    print_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = get_split_data()
