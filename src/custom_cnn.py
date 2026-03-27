# =========================================================================
# CUSTOM CNN MODEL (Özgün CNN Ağı) - v2
# =========================================================================
# İleri düzey mimari: Residual + SE Attention + Multi-Scale Feature Fusion
#
# Tasarım Felsefesi:
# - Küçük veri (200 img) → Parametre sayısını kontrol altında tut (~1.5M)
# - Residual bağlantılar → Gradient flow iyileştirir, derin ağ eğitilebilir
# - SE (Squeeze-and-Excitation) → Kanal bazlı attention, önemli feature'ları
#   vurgular (medikal görüntüde kritik: kanama bölgesi vs arka plan)
# - Multi-Scale Fusion → Farklı çözünürlüklerdeki bilgiyi birleştirir
#   (küçük kanamalar + büyük kanamalar aynı anda yakalanır)
# - Stochastic Depth → Eğitim sırasında rastgele blok atlama (regularization)
#
# Referans: SE-Net (Hu et al., 2018), ResNet (He et al., 2016)
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block - Kanal Bazlı Attention.

    Her kanalın önemini öğrenir:
    1. Squeeze: Global Average Pool → kanal bazlı istatistik
    2. Excitation: FC → ReLU → FC → Sigmoid → kanal ağırlıkları
    3. Scale: Orijinal feature map * ağırlıklar

    Medikal görüntüde neden önemli:
    → Model hangi feature kanallarının (kenar, doku, yoğunluk)
      hemorrhage tespiti için kritik olduğunu ÖĞRENIR.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    """
    Residual Block + SE Attention.

    Yapı:
        input → Conv→BN→ReLU → Conv→BN → SE → (+input) → ReLU → Dropout
                ↓                                ↑
                └── [1x1 Conv shortcut] ─────────┘  (kanal sayısı değişirse)

    Residual bağlantı neden kritik:
    → Gradient vanishing problemi çözer
    → Derin ağlar eğitilebilir hale gelir
    → "En kötü ihtimal identity mapping öğren" → zarar vermez
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.15):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(p=dropout)

        # Shortcut: kanal veya boyut değiştiğinde 1x1 conv
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)

        out = out + identity  # Residual connection
        out = F.relu(out, inplace=True)
        return out


class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Feature Extraction - Farklı Çözünürlükleri Birleştir.

    3 paralel yol:
    - 1x1 conv: Nokta bazlı (yoğunluk farkları)
    - 3x3 conv: Yerel özellikler (kenarlar, dokular)
    - 5x5 conv (3x3+3x3): Daha geniş alan (büyük yapılar)

    Neden: CT'de kanama hem küçük (subdural) hem büyük (intracerebral)
    olabilir → farklı ölçeklerde bakmalıyız.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        # Kalan kanallar (3'e tam bölünmezse)
        remaining = out_ch - 3 * branch_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * branch_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ) if remaining > 0 else nn.Identity()

        self._out_ch = out_ch
        self._branch_ch = branch_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        if isinstance(self.fuse, nn.Identity):
            return out
        return self.fuse(out)


class CustomCNN(nn.Module):
    """
    Özgün CNN Mimarisi v2 - Head CT Hemorrhage Sınıflandırma

    ┌──────────────────────────────────────────────────────────────────┐
    │ STEM: Conv 7x7 stride=2 → BN → ReLU → MaxPool   (224 → 56)   │
    ├──────────────────────────────────────────────────────────────────┤
    │ Multi-Scale Block: Farklı çözünürlükleri birleştir (56 × 48)   │
    ├──────────────────────────────────────────────────────────────────┤
    │ ResidualSE Block 1: 48 → 64   stride=2  (56 → 28)             │
    │ ResidualSE Block 2: 64 → 128  stride=2  (28 → 14)             │
    │ ResidualSE Block 3: 128 → 256 stride=2  (14 → 7)              │
    ├──────────────────────────────────────────────────────────────────┤
    │ Global Average Pooling → 256                                    │
    │ FC: 256 → 128 → BN → ReLU → Dropout(0.4) → 2                  │
    └──────────────────────────────────────────────────────────────────┘

    Yaratıcı özellikler (standart CNN'den farkları):
    1. Stem (büyük 7x7 kernel) → CT görüntüsünde geniş alandan ilk bakış
    2. Multi-Scale Block → Farklı boyut kanama tespiti
    3. SE Attention → Hangi feature kanalları önemli
    4. Residual → Derin ağ eğitilebilirlik garantisi
    5. Classifier'da BatchNorm → Son katmanda da stabilizasyon
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        # Stem: Büyük reseptif alan ile başla
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # 224 → 56

        # Multi-Scale Feature Extraction
        self.multi_scale = MultiScaleBlock(32, 48)

        # Residual SE Blocks (progressive kanal artışı)
        self.block1 = ResidualSEBlock(48, 64, stride=2, dropout=0.1)    # 56→28
        self.block2 = ResidualSEBlock(64, 128, stride=2, dropout=0.15)  # 28→14
        self.block3 = ResidualSEBlock(128, 256, stride=2, dropout=0.2)  # 14→7

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (BatchNorm dahil - daha stabil)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

        # Weight initialization (Kaiming - ReLU için optimal)
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialization - ReLU ağlar için en iyi başlangıç."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)           # (B, 32, 56, 56)
        x = self.multi_scale(x)    # (B, 48, 56, 56)
        x = self.block1(x)         # (B, 64, 28, 28)
        x = self.block2(x)         # (B, 128, 14, 14)
        x = self.block3(x)         # (B, 256, 7, 7)
        x = self.global_pool(x)    # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.classifier(x)     # (B, num_classes)
        return x


def get_custom_cnn(num_classes: int = NUM_CLASSES) -> CustomCNN:
    """Custom CNN modeli oluştur."""
    model = CustomCNN(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] CustomCNN v2 oluşturuldu (Residual + SE + MultiScale)")
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
