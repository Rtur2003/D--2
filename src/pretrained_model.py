# =========================================================================
# PRE-TRAINED MODEL (Önceden Eğitilmiş CNN)
# =========================================================================
# ConvNeXt-Tiny - Transfer Learning
#
# KURAL (Ders Notu Bölüm 2.2):
# - Küçük veri seti → Transfer learning tercih edilmeli
# - Pre-trained ağırlıklar kullanılmalı
# - Son katman(lar) fine-tune edilmeli
# =========================================================================

import torch
import torch.nn as nn
import timm
from config import NUM_CLASSES


def get_convnext_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    ConvNeXt-Tiny modeli (ImageNet pre-trained).

    Transfer Learning Stratejisi:
    1. ImageNet ağırlıkları yüklenir
    2. Son sınıflandırma katmanı değiştirilir (1000 → 2)
    3. Opsiyonel: Backbone dondurulabilir (feature extraction)
       veya tamamı fine-tune edilebilir

    Küçük veri seti için fine-tuning önerisi:
    - İlk birkaç epoch backbone dondur, sadece classifier eğit
    - Sonra tüm ağı küçük lr ile fine-tune et
    """
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes
    )

    if freeze_backbone:
        # Backbone'u dondur - sadece classifier eğitilir
        for name, param in model.named_parameters():
            if "head" not in name:  # 'head' ConvNeXt'in classifier katmanı
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[MODEL] ConvNeXt-Tiny oluşturuldu (pretrained={pretrained})")
    print(f"[MODEL] Toplam parametre: {total_params:,}")
    print(f"[MODEL] Eğitilebilir parametre: {trainable_params:,}")

    if freeze_backbone:
        print(f"[MODEL] Backbone donduruldu - sadece classifier eğitilecek")

    return model


def unfreeze_model(model: nn.Module) -> None:
    """Tüm parametreleri eğitilebilir yap (fine-tuning için)."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Tüm parametreler açıldı: {trainable:,} eğitilebilir")


if __name__ == "__main__":
    model = get_convnext_model(pretrained=True, freeze_backbone=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"[TEST] Input shape: {dummy_input.shape}")
    print(f"[TEST] Output shape: {output.shape}")
