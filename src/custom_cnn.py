# =========================================================================
# CUSTOM CNN MODEL v2 (Özgün CNN Ağı)
# =========================================================================
# Mimari: Residual + CBAM (Channel+Spatial Attention) + DropPath + MultiScale
#
# v1'den farklar:
# - SE  →  CBAM (kanal VE uzamsal dikkat): kanama bölgesi lokalizasyonu
# - DropPath (stochastic depth): gerçekten uygulanan regularization
# - Block4: dilated depthwise-separable conv → daha geniş alıcı alan
# - GELU aktivasyon: daha yumuşak gradyan, daha iyi genelleme
# - Classifier dropout 0.4 → 0.5
#
# Referans: CBAM (Woo et al., 2018), DropPath (Huang et al., 2016)
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


# ── DropPath (Stochastic Depth) ──────────────────────────────────────────
class DropPath(nn.Module):
    """
    Eğitim sırasında rastgele tüm bloğu atla (identity'e düşür).
    Farklı derinliklerde farklı drop_prob: sığ bloklar daha az atlanır.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x / keep * noise


# ── Channel Attention (SE) ───────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        w   = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * w


# ── Spatial Attention ────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Hangi KONUMLARA bakılmalı: avg+max channel map'lerinden 2→1 conv ile karar.
    Kanama bölgelerini lokalize etmek için kritik.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


# ── CBAM = Channel + Spatial Attention ──────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ── Residual Block + CBAM + DropPath ────────────────────────────────────
class ResidualCBAMBlock(nn.Module):
    """
    Yapı:
        input → Conv→BN→GELU → Conv→BN → CBAM → DropPath → (+shortcut) → GELU

    GELU: ReLU'dan daha yumuşak, daha iyi genelleme
    CBAM: Hem kanal hem uzamsal dikkat → kanama lokalizasyonu
    DropPath: Blok seviyesi regularization (her blok bağımsız feature öğrenir)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.cbam  = CBAM(out_ch)
        self.drop2d = nn.Dropout2d(p=dropout)
        self.drop_path = DropPath(drop_path)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop2d(out)
        out = self.cbam(out)
        out = self.drop_path(out) + identity
        return F.gelu(out)


# ── Depthwise Separable Dilated Block (Block4) ───────────────────────────
class DilatedDSBlock(nn.Module):
    """
    Depthwise separable + dilation=2 conv.
    7x7 feature map'te 11x11 efektif alıcı alan sağlar (global CT yapısı).
    Param sayısı: 256*9 + 256*256 ≈ 68K (regular conv'ın ~10x altında).
    """
    def __init__(self, channels: int, drop_path: float = 0.15):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn(self.pw(self.dw(x))))
        out = self.cbam(out)
        return self.drop_path(out) + x


# ── Multi-Scale Feature Extraction ──────────────────────────────────────
class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b = out_ch // 3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b), nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.GELU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.GELU(),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.GELU(),
        )
        remaining = out_ch - 3 * b
        self.fuse = (
            nn.Sequential(
                nn.Conv2d(3 * b, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            )
            if remaining > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)
        return self.fuse(out) if not isinstance(self.fuse, nn.Identity) else out


# ── CustomCNN v2 ─────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    Özgün CNN v2 - Head CT Hemorrhage Sınıflandırma

    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEM:   Conv 7x7 s=2 → BN → GELU → MaxPool        (224→56)        │
    ├──────────────────────────────────────────────────────────────────────┤
    │ MultiScale: 32→48  (1x1 + 3x3 + 5x5 paralel)                       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Block1: ResidualCBAM  48→64   stride=2  drop_path=0.00  (56→28)    │
    │ Block2: ResidualCBAM  64→128  stride=2  drop_path=0.05  (28→14)    │
    │ Block3: ResidualCBAM 128→256  stride=2  drop_path=0.10  (14→7)     │
    │ Block4: DilatedDS    256→256  dilation=2 drop_path=0.15  (7→7)     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Global Average Pool → 256                                            │
    │ Classifier: 256→128 (LayerNorm→GELU→Drop0.5) → 2                   │
    └──────────────────────────────────────────────────────────────────────┘

    Parametre: ~1.45M  (küçük veri için uygun)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # → (B, 32, 56, 56)

        self.multi_scale = MultiScaleBlock(32, 48)

        self.block1 = ResidualCBAMBlock(48,  64,  stride=2, dropout=0.05, drop_path=0.00)
        self.block2 = ResidualCBAMBlock(64,  128, stride=2, dropout=0.10, drop_path=0.05)
        self.block3 = ResidualCBAMBlock(128, 256, stride=2, dropout=0.15, drop_path=0.10)
        self.block4 = DilatedDSBlock(256, drop_path=0.15)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.multi_scale(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


def get_custom_cnn(num_classes: int = NUM_CLASSES) -> CustomCNN:
    model = CustomCNN(num_classes=num_classes)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[MODEL] CustomCNN olusturuldu (Residual + CBAM + DropPath + DilatedDS)")
    print(f"[MODEL] Toplam parametre: {total:,}")
    print(f"[MODEL] Eğitilebilir parametre: {trainable:,}")
    return model


if __name__ == "__main__":
    model = get_custom_cnn()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"[TEST] Input: {dummy.shape}  Output: {out.shape}")
    print(f"[TEST] Output: {out}")
