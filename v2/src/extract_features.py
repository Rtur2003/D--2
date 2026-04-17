"""Penultimate-layer embedding CSV (hoca: feature CSV teslim kriteri)."""

from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DEVICE,
    MODELS_DIR,
    RESULTS_DIR,
    CLASS_NAMES,
    DATA_DIR,
)
from data_preprocessing import get_transforms  # noqa: E402
from data_split import get_split_data  # noqa: E402
from pretrained_model import get_convnext_model  # noqa: E402
from custom_cnn import get_custom_cnn  # noqa: E402


def load_stats():
    with open(MODELS_DIR / "train_stats.json", "r") as f:
        d = json.load(f)
    return d["mean"], d["std"]


@torch.no_grad()
def extract_convnext(model, tensors):
    model.eval()
    feats = model.forward_features(tensors.to(DEVICE))
    pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
    return pooled.cpu().numpy()


@torch.no_grad()
def extract_custom(model, tensors):
    model.eval()
    x = tensors.to(DEVICE)
    x = model.stem(x)
    x = model.multi_scale(x)
    x = model.block1(x)
    x = model.block2(x)
    x = model.block3(x)
    x = model.global_pool(x).flatten(1)
    return x.cpu().numpy()


def load_ckpt(model, path: Path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(DEVICE)


def write_csv(out_path: Path, filenames, labels, feats):
    n = feats.shape[1]
    cols = ["filename", "label"] + [f"f{i:03d}" for i in range(n)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fn, lab, vec in zip(filenames, labels, feats):
            w.writerow([fn, lab] + [f"{v:.6f}" for v in vec])
    print(f"[WROTE] {out_path.name} ({len(filenames)}x{n+2})")


def main():
    _, _, test_df = get_split_data()
    mean, std = load_stats()
    tf = get_transforms(mean=mean, std=std, is_train=False, augment=False)

    paths = test_df["image_path"].tolist()
    filenames = [Path(p).name for p in paths]
    labels = test_df["label"].tolist()

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(tf(img))
    batch = torch.stack(tensors)

    conv = load_ckpt(get_convnext_model(), MODELS_DIR / "convnext_tiny_best.pth")
    conv_feats = extract_convnext(conv, batch)
    write_csv(
        RESULTS_DIR / "features_convnext_test.csv",
        filenames, labels, conv_feats,
    )

    cust = load_ckpt(get_custom_cnn(), MODELS_DIR / "custom_cnn_best.pth")
    cust_feats = extract_custom(cust, batch)
    write_csv(
        RESULTS_DIR / "features_custom_cnn_test.csv",
        filenames, labels, cust_feats,
    )

    combined = np.concatenate([conv_feats, cust_feats], axis=1)
    write_csv(
        RESULTS_DIR / "features_test.csv",
        filenames, labels, combined,
    )

    print("Sinif adlari:", CLASS_NAMES)
    print("label=0 -> Normal, label=1 -> Hemorrhage")


if __name__ == "__main__":
    main()
