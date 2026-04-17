# =========================================================================
# GRAD-CAM - Gradient-weighted Class Activation Mapping
# =========================================================================
# Model hangi bölgeye bakarak "hemorrhage" veya "normal" diyor?
# CT görüntüsü üzerinde ısı haritası ile gösterir.
#
# Bu, medikal AI projelerinde ZORUNLU bir adımdır:
# → Doktor modele güvenebilir mi?
# → Model gerçekten kanama bölgesine mi bakıyor?
# → Yoksa artifakt'a mı takılmış?
#
# Teknik: Son conv katmanının gradient'lerini kullanarak
#         her feature map'in önemini hesaplar.
# =========================================================================

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple
from pathlib import Path

from config import CLASS_NAMES, IMG_SIZE, DEVICE, RESULTS_DIR


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.

    Çalışma prensibi:
    1. Forward pass → hedef sınıfın skorunu al
    2. Backward pass → son conv katmanına akan gradient'leri yakala
    3. Her feature map'in gradient ortalamasını al (önem ağırlığı)
    4. Ağırlıklı toplam → sınıf aktivasyon haritası
    5. ReLU → sadece pozitif etki (negatif bölgeleri at)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook'ları kaydet
        self._register_hooks()

    def _register_hooks(self):
        """Forward ve backward hook'ları kaydet."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Grad-CAM ısı haritası üret.

        Args:
            input_tensor: (1, 3, H, W) şeklinde normalize edilmiş görüntü
            target_class: Hedef sınıf (None → en yüksek skor)

        Returns:
            cam: (H, W) şeklinde 0-1 arası ısı haritası
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward (hedef sınıfın skoru)
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Grad-CAM hesaplama
        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]      # (C, H, W)

        # Her kanalın gradient ortalaması → önem ağırlığı
        weights = gradients.mean(dim=(1, 2))   # (C,)

        # Ağırlıklı toplam
        cam = torch.zeros(activations.shape[1:], device=DEVICE)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU → sadece pozitif etkiler
        cam = F.relu(cam)

        # Normalize (0-1)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()

        # Orijinal görüntü boyutuna resize
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
        ) / 255.0

        return cam_resized, probs[0].detach().cpu().numpy(), target_class


def get_target_layer(model, model_name: str):
    """Model tipine göre Grad-CAM için hedef katmanı bul."""
    if "convnext" in model_name.lower():
        # ConvNeXt-Tiny: son stage'in son bloğu
        return model.stages[-1].blocks[-1]
    else:
        # Custom CNN v2: son residual bloğun son conv katmanı
        return model.block3.conv2


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """CAM ısı haritasını orijinal görüntü üzerine bindir."""
    # Jet colormap uygula
    heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3) - RGB

    # Overlay
    if image.max() > 1:
        image = image / 255.0

    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def visualize_gradcam_grid(
    model,
    model_name: str,
    image_paths: list,
    labels: list,
    transform,
    save_path: str = None,
    num_samples: int = 8
) -> None:
    """
    Birden fazla görüntü için Grad-CAM grid gösterimi.

    4 Normal + 4 Hemorrhage örneği:
    - Üst satır: Orijinal görüntü
    - Alt satır: Grad-CAM overlay
    """
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Her sınıftan eşit sayıda seç
    normal_idx = [i for i, l in enumerate(labels) if l == 0][:num_samples // 2]
    hemorrhage_idx = [i for i, l in enumerate(labels) if l == 1][:num_samples // 2]
    selected_idx = normal_idx + hemorrhage_idx

    n_cols = len(selected_idx)
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 10))

    fig.suptitle(
        f"Grad-CAM Analizi - {model_name}\n"
        f"Model hangi bölgeye bakarak karar veriyor?",
        fontsize=14, fontweight="bold", y=1.02
    )

    for col, idx in enumerate(selected_idx):
        img_path = image_paths[idx]
        true_label = labels[idx]

        # Orijinal görüntü
        original = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        original_np = np.array(original) / 255.0

        # Grad-CAM
        input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        cam, probs, pred_class = grad_cam.generate(input_tensor)

        # Overlay
        overlay = overlay_cam_on_image(original_np, cam, alpha=0.5)

        # Orijinal
        axes[0, col].imshow(original_np)
        axes[0, col].set_title(
            f"Gerçek: {CLASS_NAMES[true_label]}",
            fontsize=9, fontweight="bold",
            color="green" if true_label == 0 else "red"
        )
        axes[0, col].axis("off")

        # CAM heatmap
        axes[1, col].imshow(cam, cmap="jet")
        axes[1, col].set_title("Aktivasyon Haritası", fontsize=8)
        axes[1, col].axis("off")

        # Overlay
        axes[2, col].imshow(overlay)
        pred_text = f"Tahmin: {CLASS_NAMES[pred_class]}\n({probs[pred_class]:.1%})"
        correct = pred_class == true_label
        axes[2, col].set_title(
            pred_text, fontsize=9, fontweight="bold",
            color="green" if correct else "red"
        )
        axes[2, col].axis("off")

    # Satır başlıkları
    axes[0, 0].set_ylabel("Orijinal CT", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("Overlay", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAD-CAM] Kaydedildi: {save_path}")

    plt.close()


def visualize_single_gradcam(
    model,
    model_name: str,
    image_path: str,
    transform,
    save_path: str = None
) -> Tuple[str, float]:
    """Tek görüntü için detaylı Grad-CAM analizi."""
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    original = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original) / 255.0

    input_tensor = transform(original).unsqueeze(0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Her sınıf için CAM
    for cls_idx in range(2):
        cam, probs, _ = grad_cam.generate(input_tensor, target_class=cls_idx)
        overlay = overlay_cam_on_image(original_np, cam, alpha=0.5)

        axes[cls_idx + 1].imshow(overlay)
        axes[cls_idx + 1].set_title(
            f"Grad-CAM: {CLASS_NAMES[cls_idx]}\n(Skor: {probs[cls_idx]:.3f})",
            fontsize=11, fontweight="bold"
        )
        axes[cls_idx + 1].axis("off")

    # Orijinal
    axes[0].imshow(original_np)
    axes[0].set_title("Orijinal CT", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Fark haritası
    cam_h, probs_h, _ = grad_cam.generate(input_tensor, target_class=1)
    cam_n, _, _ = grad_cam.generate(input_tensor, target_class=0)
    diff = cam_h - cam_n
    axes[3].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_title(
        "Fark (Hemorrhage - Normal)\nKırmızı=Kanama bölgesi",
        fontsize=10, fontweight="bold"
    )
    axes[3].axis("off")

    pred_class = CLASS_NAMES[probs_h.argmax()]
    confidence = probs_h.max()
    fig.suptitle(
        f"{model_name} | Tahmin: {pred_class} ({confidence:.1%})",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAD-CAM] Kaydedildi: {save_path}")

    plt.close()

    return pred_class, confidence


if __name__ == "__main__":
    print("[GRAD-CAM] Bu modül eğitimden sonra çalıştırılmalıdır.")
    print("[GRAD-CAM] Kullanım: python main.py --all sonrası otomatik çalışır.")
