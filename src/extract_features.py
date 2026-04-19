# =========================================================================
# FEATURE EXTRACTION (Ozellik Cikarma)
# =========================================================================
# Hoca dosya degerlendirme kriteri: "feature CSV dosyasi" zorunlu.
#
# Bu modul her iki modelin son katman oncesi (penultimate layer) feature
# vektorlerini cikarir ve tek bir CSV dosyasina kaydeder.
#
# CSV iceriği:
#   image_id, split, true_label, true_label_name,
#   convnext_pred, convnext_conf, convnext_prob_normal, convnext_prob_hemorrhage,
#   custom_pred, custom_conf, custom_prob_normal, custom_prob_hemorrhage,
#   convnext_f000..convnext_f767  (768-d feature),
#   custom_f000..custom_f255      (256-d feature)
# =========================================================================

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE
)
from data_split import get_split_data
from data_preprocessing import HeadCTDataset, get_transforms
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model


class _FeatureHook:
    """Penultimate-layer feature yakalamak icin forward hook."""

    def __init__(self, module: torch.nn.Module):
        self.feat = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, _inp, out):
        self.feat = out.detach().flatten(1).cpu().numpy()

    def remove(self):
        self.handle.remove()


def _load_models():
    """Egitilmis modelleri yukler ve eval moduna alir."""
    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(
        str(MODELS_DIR / "convnext_tiny_best.pth"),
        map_location=DEVICE, weights_only=False
    )
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)

    custom = get_custom_cnn()
    ckpt = torch.load(
        str(MODELS_DIR / "custom_cnn_best.pth"),
        map_location=DEVICE, weights_only=False
    )
    custom.load_state_dict(ckpt["model_state_dict"])
    custom.eval().to(DEVICE)

    return convnext, custom


def _all_split_dfs():
    """Tum split'leri (train/val/test) tek dataframe'de birlestir."""
    train_df, val_df, test_df = get_split_data()
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def extract_features_to_csv(output_path: Path = None) -> Path:
    """
    Tum 200 goruntu icin feature + tahminleri CSV'ye yazar.

    Returns:
        CSV dosyasinin tam yolu.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "features.csv"

    # Normalizasyon istatistikleri
    with open(str(MODELS_DIR / "train_stats.json"), "r") as f:
        stats = json.load(f)

    transform = get_transforms(stats["mean"], stats["std"], is_train=False)

    # Veri
    df = _all_split_dfs()
    paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    dataset = HeadCTDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Modeller
    convnext, custom = _load_models()

    # Penultimate-layer hook'lari
    convnext_hook = _FeatureHook(convnext.head.global_pool)  # (B, 768)
    custom_hook = _FeatureHook(custom.global_pool)            # (B, 256, 1, 1)

    convnext_feats: List[np.ndarray] = []
    custom_feats: List[np.ndarray] = []
    convnext_probs: List[np.ndarray] = []
    custom_probs: List[np.ndarray] = []

    print(f"[FEATURES] {len(dataset)} goruntu icin feature cikariliyor...")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(DEVICE)

            logits1 = convnext(images)
            logits2 = custom(images)

            convnext_probs.append(F.softmax(logits1, dim=1).cpu().numpy())
            custom_probs.append(F.softmax(logits2, dim=1).cpu().numpy())

            convnext_feats.append(convnext_hook.feat)
            custom_feats.append(custom_hook.feat)

    convnext_hook.remove()
    custom_hook.remove()

    convnext_feats = np.vstack(convnext_feats)   # (N, 768)
    custom_feats = np.vstack(custom_feats)       # (N, 256)
    convnext_probs = np.vstack(convnext_probs)   # (N, 2)
    custom_probs = np.vstack(custom_probs)       # (N, 2)

    # Tahminler
    convnext_pred = convnext_probs.argmax(axis=1)
    custom_pred = custom_probs.argmax(axis=1)

    # Meta + tahmin sutunlari
    meta = pd.DataFrame({
        "image_id": [Path(p).stem for p in paths],
        "split": df["split"].values,
        "true_label": labels,
        "true_label_name": [CLASS_NAMES[l] for l in labels],

        "convnext_pred": [CLASS_NAMES[p] for p in convnext_pred],
        "convnext_conf": convnext_probs.max(axis=1),
        "convnext_prob_normal": convnext_probs[:, 0],
        "convnext_prob_hemorrhage": convnext_probs[:, 1],

        "custom_pred": [CLASS_NAMES[p] for p in custom_pred],
        "custom_conf": custom_probs.max(axis=1),
        "custom_prob_normal": custom_probs[:, 0],
        "custom_prob_hemorrhage": custom_probs[:, 1],
    })

    # Feature sutunlari
    convnext_cols = pd.DataFrame(
        convnext_feats,
        columns=[f"convnext_f{i:03d}" for i in range(convnext_feats.shape[1])],
    )
    custom_cols = pd.DataFrame(
        custom_feats,
        columns=[f"custom_f{i:03d}" for i in range(custom_feats.shape[1])],
    )

    out_df = pd.concat([meta, convnext_cols, custom_cols], axis=1)
    out_df.to_csv(output_path, index=False, float_format="%.6f")

    print(f"[FEATURES] Yazildi: {output_path}")
    print(f"[FEATURES] Boyut: {out_df.shape[0]} satir x {out_df.shape[1]} sutun")
    print(f"[FEATURES]   - ConvNeXt feature boyutu: {convnext_feats.shape[1]}")
    print(f"[FEATURES]   - Custom CNN feature boyutu: {custom_feats.shape[1]}")

    # Ozet versiyon (sadece meta + tahminler, feature yok)
    summary_path = RESULTS_DIR / "predictions_summary.csv"
    meta.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"[FEATURES] Ozet (sadece tahminler): {summary_path}")

    return output_path


if __name__ == "__main__":
    extract_features_to_csv()
