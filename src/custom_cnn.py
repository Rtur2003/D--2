# =========================================================================
# CUSTOM CNN MODEL (Özgün CNN Ağı)
# =========================================================================
# Öğrenci tarafından tasarlanan özgün CNN mimarisi.
#
# Mimari Tasarım Kararları:
# - Küçük veri seti (200 görüntü) → Fazla parametre overfitting yapar
# - BatchNorm + Dropout ile regularization
# - Global Average Pooling (parametre azaltma)
# - Progressive kanal artışı (32 → 64 → 128 → 256)
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Tekrar eden Convolution bloğu: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        return x


class CustomCNN(nn.Module):
    """
    Özgün CNN Mimarisi - Head CT Hemorrhage Sınıflandırma

    Mimari:
    ┌─────────────────────────────────────────┐
    │ Input: 224×224×3                        │
    ├─────────────────────────────────────────┤
    │ ConvBlock 1: 3 → 32 kanal (112×112)     │
    │ ConvBlock 2: 32 → 64 kanal (56×56)      │
    │ ConvBlock 3: 64 → 128 kanal (28×28)     │
    │ ConvBlock 4: 128 → 256 kanal (14×14)    │
    ├─────────────────────────────────────────┤
    │ Global Average Pooling → 256            │
    │ FC: 256 → 128 → Dropout → 2            │
    └─────────────────────────────────────────┘

    Toplam parametre: ~1.2M (küçük veri için uygun boyut)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        # Feature extractor
        self.block1 = ConvBlock(3, 32, dropout=0.1)
        self.block2 = ConvBlock(32, 64, dropout=0.15)
        self.block3 = ConvBlock(64, 128, dropout=0.2)
        self.block4 = ConvBlock(128, 256, dropout=0.25)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # 224→112
        x = self.block2(x)   # 112→56
        x = self.block3(x)   # 56→28
        x = self.block4(x)   # 28→14

        x = self.global_pool(x)  # 14→1
        x = x.view(x.size(0), -1)  # Flatten: (B, 256)
        x = self.classifier(x)     # (B, num_classes)

        return x


def get_custom_cnn(num_classes: int = NUM_CLASSES) -> CustomCNN:
    """Custom CNN modeli oluştur."""
    model = CustomCNN(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] CustomCNN oluşturuldu")
    print(f"[MODEL] Toplam parametre: {total_params:,}")
    print(f"[MODEL] Eğitilebilir parametre: {trainable_params:,}")
    return model


if __name__ == "__main__":
    model = get_custom_cnn()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"[TEST] Input shape: {dummy_input.shape}")
    print(f"[TEST] Output shape: {output.shape}")
    print(f"[TEST] Output: {output}")
