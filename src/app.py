# =========================================================================
# ARAYÜZ (Interface) - Gradio
# =========================================================================
# Profesyonel medikal AI arayüzü:
# - Dosya seçimi ile CT görüntüsü yükleme
# - 3 model seçeneği: ConvNeXt, Custom CNN, Ensemble
# - Olasılık skorları ile tahmin
# - Grad-CAM ısı haritası (model neye bakıyor?)
# - Her iki modelin ayrı ayrı skorları (ensemble detay)
# =========================================================================

import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gradio as gr

from config import DEVICE, MODELS_DIR, CLASS_NAMES, IMG_SIZE
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model
from ensemble import EnsembleModel
from gradcam import GradCAM, get_target_layer, overlay_cam_on_image


# ── Global model cache ──────────────────────────────────────────────────

_models_cache = {}


def _get_stats():
    """Train normalizasyon istatistiklerini yükle."""
    stats_path = MODELS_DIR / "train_stats.json"
    with open(str(stats_path), "r") as f:
        return json.load(f)


def _get_transform(stats):
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


def _load_models():
    """Modelleri yükle ve cache'e al."""
    if "convnext" in _models_cache:
        return

    stats = _get_stats()

    # ConvNeXt
    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(str(MODELS_DIR / "convnext_tiny_best.pth"), map_location=DEVICE, weights_only=False)
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)

    # Custom CNN
    custom = get_custom_cnn()
    ckpt = torch.load(str(MODELS_DIR / "custom_cnn_best.pth"), map_location=DEVICE, weights_only=False)
    custom.load_state_dict(ckpt["model_state_dict"])
    custom.eval().to(DEVICE)

    # Ensemble
    ensemble = EnsembleModel(convnext, custom, weight1=0.5, weight2=0.5)

    _models_cache["convnext"] = convnext
    _models_cache["custom"] = custom
    _models_cache["ensemble"] = ensemble
    _models_cache["stats"] = stats
    _models_cache["transform"] = _get_transform(stats)


# ── Ana Tahmin Fonksiyonu ───────────────────────────────────────────────

def predict_with_gradcam(image, model_choice: str):
    """
    Görüntüyü sınıflandır + Grad-CAM ısı haritası üret.

    Returns:
        label_output: Olasılık skorları
        gradcam_image: Grad-CAM overlay görüntüsü
        detail_text: Detaylı rapor
    """
    if image is None:
        return {}, None, "Lütfen bir görüntü yükleyin."

    _load_models()
    transform = _models_cache["transform"]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Model seçimi
    if "Ensemble" in model_choice:
        ensemble = _models_cache["ensemble"]
        result = ensemble.predict_single(input_tensor)
        scores = result["ensemble_scores"]

        detail_lines = [
            f"=== ENSEMBLE TAHMİN RAPORU ===",
            f"Sonuç: {result['prediction']} ({result['confidence']:.1%})",
            f"",
            f"--- ConvNeXt-Tiny Skorları ---",
        ]
        for cls, score in result["model1_scores"].items():
            detail_lines.append(f"  {cls}: {score:.4f} ({score:.1%})")
        detail_lines.append(f"")
        detail_lines.append(f"--- Custom CNN Skorları ---")
        for cls, score in result["model2_scores"].items():
            detail_lines.append(f"  {cls}: {score:.4f} ({score:.1%})")
        detail_lines.append(f"")
        detail_lines.append(f"--- Ensemble Skorları ---")
        for cls, score in result["ensemble_scores"].items():
            detail_lines.append(f"  {cls}: {score:.4f} ({score:.1%})")

        detail_text = "\n".join(detail_lines)

        # Grad-CAM: ConvNeXt üzerinden (daha güvenilir)
        model_for_cam = _models_cache["convnext"]
        model_name_for_cam = "convnext"
    else:
        if "ConvNeXt" in model_choice:
            model = _models_cache["convnext"]
            model_name_for_cam = "convnext"
        else:
            model = _models_cache["custom"]
            model_name_for_cam = "custom"

        model_for_cam = model

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        pred = CLASS_NAMES[probs.argmax()]
        conf = probs.max()

        detail_text = (
            f"=== TAHMİN RAPORU ===\n"
            f"Model: {model_choice}\n"
            f"Sonuç: {pred} ({conf:.1%})\n\n"
            f"Skorlar:\n"
        )
        for cls, score in scores.items():
            detail_text += f"  {cls}: {score:.4f} ({score:.1%})\n"

    # Grad-CAM
    try:
        target_layer = get_target_layer(model_for_cam, model_name_for_cam)
        grad_cam = GradCAM(model_for_cam, target_layer)

        original_np = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        cam, _, pred_class = grad_cam.generate(
            transform(image).unsqueeze(0)
        )
        overlay = overlay_cam_on_image(original_np, cam, alpha=0.45)
        gradcam_image = (overlay * 255).astype(np.uint8)
    except Exception as e:
        gradcam_image = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        detail_text += f"\n[Grad-CAM hatası: {e}]"

    return scores, gradcam_image, detail_text


# ── Gradio Arayüzü ─────────────────────────────────────────────────────

def create_interface():
    """Profesyonel Gradio arayüzü."""

    with gr.Blocks(
        title="Head CT Hemorrhage Classifier",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red")
    ) as demo:

        gr.Markdown("""
        # Head CT Hemorrhage Siniflandirma
        ### BM 480 Derin Ogrenme - Proje 2

        Bir Head CT goruntusu yukleyin ve beyin kanamasi (hemorrhage) olup
        olmadigini yapay zeka ile tespit edin.

        **3 Model Secenegi:**
        - **ConvNeXt-Tiny**: ImageNet pre-trained transfer learning modeli
        - **Custom CNN**: Ogrenci tarafindan tasarlanan ozgun mimari
        - **Ensemble**: Iki modelin birlesimi (en guvenilir sonuc)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="CT Goruntusu Yukle",
                    type="pil",
                    height=350
                )
                model_choice = gr.Radio(
                    choices=[
                        "ConvNeXt-Tiny (Pre-trained)",
                        "Custom CNN (Ozgun)",
                        "Ensemble (Birlesik - Onerilen)"
                    ],
                    value="Ensemble (Birlesik - Onerilen)",
                    label="Model Secimi"
                )
                predict_btn = gr.Button(
                    "Tahmin Et",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Olasilik Skorlari",
                    num_top_classes=2
                )
                gradcam_output = gr.Image(
                    label="Grad-CAM (Model neye bakiyor?)",
                    height=300
                )

        with gr.Row():
            detail_output = gr.Textbox(
                label="Detayli Rapor",
                lines=12,
                interactive=False
            )

        predict_btn.click(
            fn=predict_with_gradcam,
            inputs=[image_input, model_choice],
            outputs=[output_label, gradcam_output, detail_output]
        )

        gr.Markdown("""
        ---
        **Siniflar:** Normal (kanama yok) | Hemorrhage (beyin kanamasi)

        **Grad-CAM:** Isil harita modelin CT'nin hangi bolgesine bakarak karar verdigini gosterir.
        Kirmizi bolgeler = yuksek aktivasyon.

        *Not: Bu sistem egitim amaclidir. Tibbi teshis icin kullanilamaz.*
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
