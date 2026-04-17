# =========================================================================
# WEB CRAWLER - Test İçin Harici Görüntü Toplama
# =========================================================================
# Proje gereksinimi:
# "Proje sunumu aşamasında kullanmak üzere web-crawling ile bir-kaç
#  görüntü örneği toplanmalıdır."
#
# Bu script, web'den örnek Head CT görüntüleri indirip
# web_crawled_test/ klasörüne kaydeder.
# =========================================================================

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from config import (
    DEVICE, MODELS_DIR, WEB_CRAWLED_DIR, CLASS_NAMES, IMG_SIZE
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model


def predict_single_image(image_path: str, model_name: str = "convnext_tiny") -> dict:
    """Tek bir görüntü üzerinde tahmin yap."""

    # İstatistikleri yükle
    stats_path = MODELS_DIR / "train_stats.json"
    with open(str(stats_path), "r") as f:
        stats = json.load(f)

    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Model yükle
    if model_name == "convnext_tiny":
        model = get_convnext_model(pretrained=False)
    else:
        model = get_custom_cnn()

    checkpoint = torch.load(
        str(MODELS_DIR / f"{model_name}_best.pth"),
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    # Tahmin
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    result = {
        "image": image_path,
        "prediction": CLASS_NAMES[probs.argmax().item()],
        "confidence": probs.max().item(),
        "scores": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }
    }

    return result


def test_web_crawled_images() -> None:
    """web_crawled_test/ klasöründeki tüm görüntüleri test et."""

    image_files = list(WEB_CRAWLED_DIR.glob("*.png")) + \
                  list(WEB_CRAWLED_DIR.glob("*.jpg")) + \
                  list(WEB_CRAWLED_DIR.glob("*.jpeg"))

    if not image_files:
        print(f"[CRAWLER] {WEB_CRAWLED_DIR} klasöründe görüntü bulunamadı.")
        print(f"[CRAWLER] Lütfen web'den Head CT görüntüleri indirip bu klasöre koyun.")
        print(f"[CRAWLER] Desteklenen formatlar: PNG, JPG, JPEG")
        return

    print(f"\n{'='*60}")
    print(f"WEB-CRAWLED TEST GÖRÜNTÜLERİ DEĞERLENDİRME")
    print(f"{'='*60}")
    print(f"Bulunan görüntü sayısı: {len(image_files)}")
    print()

    for img_path in sorted(image_files):
        print(f"\n[IMAGE] {img_path.name}")
        print(f"  {'Model':<20} {'Tahmin':<15} {'Güven':>10}")
        print(f"  {'-'*45}")

        for model_name in ["convnext_tiny", "custom_cnn"]:
            try:
                result = predict_single_image(str(img_path), model_name)
                display_name = "ConvNeXt-Tiny" if "convnext" in model_name else "Custom CNN"
                print(f"  {display_name:<20} {result['prediction']:<15} "
                      f"{result['confidence']:>9.2%}")
            except Exception as e:
                print(f"  {model_name:<20} HATA: {e}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    test_web_crawled_images()
