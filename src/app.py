import json
import os
import socket
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (reserved for future chart exports)

import gradio as gr

from config import DEVICE, MODELS_DIR, CLASS_NAMES, IMG_SIZE, DATA_DIR
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model
from ensemble import EnsembleModel
from gradcam import GradCAM, get_target_layer, overlay_cam_on_image


_models_cache = {}
_prediction_counter = {"count": 0, "last_time": None}

MODEL_LABELS = {
    "convnext": "ConvNeXt-Tiny (Pre-trained)",
    "custom": "Custom CNN (Ozgun)",
    "ensemble": "Ensemble (Onerilen)",
}


MEDICAL_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0f172a",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1e293b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_title_text_weight="600",
    block_label_text_weight="500",
    block_radius="10px",
    button_primary_background_fill="linear-gradient(135deg, #1e3a5f 0%, #0f766e 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1e40af 0%, #0d9488 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#1e3a5f",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    button_secondary_text_color="#1e293b",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_focus="#0f766e",
    color_accent_soft="#ccfbf1",
)


CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f766e 100%);
    color: white; padding: 22px 28px; border-radius: 12px;
    margin-bottom: 18px; box-shadow: 0 4px 14px rgba(30, 58, 95, 0.18);
}
.app-header h1 { margin: 0; font-size: 1.55rem; font-weight: 600; letter-spacing: -0.01em; }
.app-header .subtitle { opacity: 0.88; font-size: 0.92rem; margin-top: 4px; }
.app-header .badge-row { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.app-header .badge {
    background: rgba(255,255,255,0.14); padding: 3px 10px;
    border-radius: 999px; font-size: 0.72rem; letter-spacing: 0.02em;
    border: 1px solid rgba(255,255,255,0.25);
}

.status-card {
    background: #ecfdf5; border: 1px solid #a7f3d0;
    padding: 10px 14px; border-radius: 8px; font-size: 0.85rem;
    color: #065f46; font-family: 'JetBrains Mono', monospace;
}
.status-card.busy { background: #fef3c7; border-color: #fcd34d; color: #92400e; }
.status-card.error { background: #fef2f2; border-color: #fecaca; color: #991b1b; }

.result-tabs .tab-nav { border-bottom: 2px solid #e2e8f0; }
.result-tabs button.selected {
    border-bottom-color: #0f766e !important;
    color: #0f766e !important; font-weight: 600;
}

.disclaimer {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 10px 14px; border-radius: 4px; margin-top: 16px;
    font-size: 0.82rem; color: #78350f;
}

.sidebar-section { margin-top: 18px; }
.sidebar-section h4 {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #64748b; font-weight: 600; margin: 0 0 8px 0;
}

.metric-row { display: flex; gap: 12px; margin-top: 8px; }
.metric-pill {
    flex: 1; background: #f1f5f9; padding: 10px 12px; border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.metric-pill .label {
    font-size: 0.7rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 3px;
}
.metric-pill .value {
    font-size: 1.1rem; font-weight: 600; color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}

.app-footer {
    margin-top: 24px; padding: 14px 18px;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    font-size: 0.78rem; color: #64748b;
}
.app-footer strong { color: #334155; }
"""


def _get_stats():
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
    if "convnext" in _models_cache:
        return

    stats = _get_stats()

    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(str(MODELS_DIR / "convnext_tiny_best.pth"), map_location=DEVICE, weights_only=False)
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)

    custom = get_custom_cnn()
    ckpt = torch.load(str(MODELS_DIR / "custom_cnn_best.pth"), map_location=DEVICE, weights_only=False)
    custom.load_state_dict(ckpt["model_state_dict"])
    custom.eval().to(DEVICE)

    ensemble = EnsembleModel(convnext, custom, weight1=0.5, weight2=0.5)

    _models_cache.update({
        "convnext": convnext,
        "custom": custom,
        "ensemble": ensemble,
        "stats": stats,
        "transform": _get_transform(stats),
    })


def _tta_forward(model, input_tensor):
    """4-view test-time augmentation: identity, hflip, vflip, 180-rot. Averages softmax."""
    views = [
        input_tensor,
        torch.flip(input_tensor, dims=[3]),
        torch.flip(input_tensor, dims=[2]),
        torch.flip(input_tensor, dims=[2, 3]),
    ]
    probs_stack = []
    with torch.no_grad():
        for v in views:
            logits = model(v)
            probs_stack.append(torch.softmax(logits, dim=1))
    return torch.stack(probs_stack, dim=0).mean(dim=0)


def _confidence_banner(pred_class: str, confidence: float) -> str:
    color = "#0f766e" if pred_class == "Normal" else "#be123c"
    icon = "Normal" if pred_class == "Normal" else "Kanama Suphesi"
    return (
        f"<div style='padding:14px 18px;border-radius:10px;"
        f"background:linear-gradient(135deg,{color} 0%,{color}cc 100%);"
        f"color:white;font-family:Inter,sans-serif;'>"
        f"<div style='font-size:0.75rem;opacity:0.85;letter-spacing:0.08em;"
        f"text-transform:uppercase;'>Sonuc</div>"
        f"<div style='font-size:1.4rem;font-weight:600;margin-top:2px;'>{icon}</div>"
        f"<div style='margin-top:8px;font-size:0.85rem;opacity:0.92;'>"
        f"Guven skoru: <b>{confidence:.1%}</b></div></div>"
    )


def _ensemble_html(result: dict) -> str:
    def _row(cls: str, s1: float, s2: float, se: float) -> str:
        return (
            f"<tr><td style='padding:8px 12px;font-weight:500;'>{cls}</td>"
            f"<td style='padding:8px 12px;text-align:right;font-family:JetBrains Mono,monospace;'>{s1:.3f}</td>"
            f"<td style='padding:8px 12px;text-align:right;font-family:JetBrains Mono,monospace;'>{s2:.3f}</td>"
            f"<td style='padding:8px 12px;text-align:right;font-family:JetBrains Mono,monospace;font-weight:600;color:#0f766e;'>{se:.3f}</td></tr>"
        )

    rows = "".join(
        _row(cls, result["model1_scores"][cls], result["model2_scores"][cls], result["ensemble_scores"][cls])
        for cls in CLASS_NAMES
    )
    return (
        "<div style='overflow:hidden;border:1px solid #e2e8f0;border-radius:10px;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>"
        "<thead><tr style='background:#f1f5f9;'>"
        "<th style='padding:10px 12px;text-align:left;font-weight:600;color:#475569;'>Sinif</th>"
        "<th style='padding:10px 12px;text-align:right;font-weight:600;color:#475569;'>ConvNeXt</th>"
        "<th style='padding:10px 12px;text-align:right;font-weight:600;color:#475569;'>Custom CNN</th>"
        "<th style='padding:10px 12px;text-align:right;font-weight:600;color:#0f766e;'>Ensemble</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _status_html(text: str, kind: str = "ready") -> str:
    cls = {"ready": "status-card", "busy": "status-card busy", "error": "status-card error"}[kind]
    return f"<div class='{cls}'>{text}</div>"


def predict(image, model_choice_label: str, use_tta: bool, show_gradcam: bool):
    """Single-entry predict; returns (banner_html, label_dict, gradcam_img, detail_text, ensemble_html, status_html)."""
    if image is None:
        empty_banner = (
            "<div style='padding:14px 18px;border-radius:10px;background:#f1f5f9;"
            "color:#64748b;font-family:Inter,sans-serif;text-align:center;'>"
            "Gorsel yukleyin veya asagidaki orneklerden birini secin."
            "</div>"
        )
        return empty_banner, {}, None, "", "", _status_html("Bekleniyor: gorsel yok.", "ready")

    try:
        _load_models()
    except Exception as e:
        return (
            f"<div style='padding:14px;color:#991b1b;'>Model yukleme hatasi: {e}</div>",
            {}, None, "", "", _status_html(f"Hata: {e}", "error"),
        )

    transform = _models_cache["transform"]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model_key = {v: k for k, v in MODEL_LABELS.items()}.get(model_choice_label, "ensemble")

    ensemble_html = ""
    if model_key == "ensemble":
        ensemble = _models_cache["ensemble"]
        if use_tta:
            m1 = _tta_forward(_models_cache["convnext"], input_tensor)[0].cpu().numpy()
            m2 = _tta_forward(_models_cache["custom"], input_tensor)[0].cpu().numpy()
            probs = 0.5 * m1 + 0.5 * m2
            result = {
                "model1_scores": {CLASS_NAMES[i]: float(m1[i]) for i in range(len(CLASS_NAMES))},
                "model2_scores": {CLASS_NAMES[i]: float(m2[i]) for i in range(len(CLASS_NAMES))},
                "ensemble_scores": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
                "prediction": CLASS_NAMES[int(probs.argmax())],
                "confidence": float(probs.max()),
            }
        else:
            result = ensemble.predict_single(input_tensor)
            probs = np.array([result["ensemble_scores"][c] for c in CLASS_NAMES])

        scores = result["ensemble_scores"]
        pred = result["prediction"]
        conf = result["confidence"]
        ensemble_html = _ensemble_html(result)

        model_for_cam = _models_cache["convnext"]
        model_name_for_cam = "convnext"
        detail_lines = [
            f"MODEL: Ensemble (ConvNeXt + Custom CNN, soft-voting)",
            f"TTA: {'Acik (4-view)' if use_tta else 'Kapali'}",
            f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Sonuc: {pred}  (guven: {conf:.2%})",
            "",
            "--- ConvNeXt-Tiny ---",
            *[f"  {c}: {result['model1_scores'][c]:.4f}" for c in CLASS_NAMES],
            "",
            "--- Custom CNN ---",
            *[f"  {c}: {result['model2_scores'][c]:.4f}" for c in CLASS_NAMES],
            "",
            "--- Ensemble (0.5/0.5) ---",
            *[f"  {c}: {result['ensemble_scores'][c]:.4f}" for c in CLASS_NAMES],
        ]
        detail_text = "\n".join(detail_lines)
    else:
        model = _models_cache[model_key]
        model_for_cam = model
        model_name_for_cam = model_key

        if use_tta:
            probs = _tta_forward(model, input_tensor)[0].cpu().numpy()
        else:
            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()

        scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        pred = CLASS_NAMES[int(probs.argmax())]
        conf = float(probs.max())

        detail_lines = [
            f"MODEL: {MODEL_LABELS[model_key]}",
            f"TTA: {'Acik (4-view)' if use_tta else 'Kapali'}",
            f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Sonuc: {pred}  (guven: {conf:.2%})",
            "",
            "--- Olasilik Skorlari ---",
            *[f"  {c}: {scores[c]:.4f}" for c in CLASS_NAMES],
        ]
        detail_text = "\n".join(detail_lines)

    gradcam_image = None
    if show_gradcam:
        try:
            target_layer = get_target_layer(model_for_cam, model_name_for_cam)
            grad_cam = GradCAM(model_for_cam, target_layer)
            original_np = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            cam, _, _ = grad_cam.generate(transform(image).unsqueeze(0))
            overlay = overlay_cam_on_image(original_np, cam, alpha=0.45)
            gradcam_image = (overlay * 255).astype(np.uint8)
        except Exception as e:
            gradcam_image = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
            detail_text += f"\n\n[Grad-CAM uretilemedi: {e}]"

    _prediction_counter["count"] += 1
    _prediction_counter["last_time"] = datetime.now().strftime("%H:%M:%S")
    status_text = (
        f"Analiz #{_prediction_counter['count']} | "
        f"{MODEL_LABELS.get(model_key, model_choice_label)} | "
        f"Son islem: {_prediction_counter['last_time']}"
    )

    banner = _confidence_banner(pred, conf)
    return banner, scores, gradcam_image, detail_text, ensemble_html, _status_html(status_text, "ready")


def _example_paths():
    """Galeri icin 6 ornek: 3 hemorrhage (001-099) + 3 normal (100-199)."""
    candidates = [
        ("001.png", "Kanama ornegi 1"),
        ("015.png", "Kanama ornegi 2"),
        ("042.png", "Kanama ornegi 3"),
        ("105.png", "Normal ornek 1"),
        ("130.png", "Normal ornek 2"),
        ("172.png", "Normal ornek 3"),
    ]
    paths = []
    for fname, _label in candidates:
        p = DATA_DIR / fname
        if p.exists():
            paths.append([str(p)])
    return paths


def create_interface():
    with gr.Blocks(theme=MEDICAL_THEME, css=CUSTOM_CSS, title="Head CT Hemorrhage Classifier") as demo:
        gr.HTML(
            "<div class='app-header'>"
            "<h1>Head CT Hemorrhage Classifier</h1>"
            "<div class='subtitle'>Beyin kanamasi tespiti icin derin ogrenme tabanli karar destek arayuzu</div>"
            "<div class='badge-row'>"
            "<span class='badge'>ConvNeXt-Tiny + Custom CNN</span>"
            "<span class='badge'>Grad-CAM Explainability</span>"
            "<span class='badge'>Test-Time Augmentation</span>"
            "<span class='badge'>Soft-Voting Ensemble</span>"
            "</div></div>"
        )

        with gr.Row():
            with gr.Sidebar(position="left", width=300, open=True):
                gr.HTML("<div class='sidebar-section'><h4>Model Secimi</h4></div>")
                model_choice = gr.Radio(
                    choices=list(MODEL_LABELS.values()),
                    value=MODEL_LABELS["ensemble"],
                    label="",
                    info="Ensemble: iki modelin soft-voting birlesimi (en guvenilir).",
                )

                gr.HTML("<div class='sidebar-section'><h4>Gelismis</h4></div>")
                use_tta = gr.Checkbox(
                    label="Test-Time Augmentation (TTA)",
                    value=True,
                    info="4 goruntu varyantinin ortalamasi; +%1-3 dogruluk.",
                )
                show_gradcam = gr.Checkbox(
                    label="Grad-CAM uret",
                    value=True,
                    info="Modelin dikkat ettigi bolgeleri isi haritasi olarak goster.",
                )

                with gr.Accordion("Model detaylari", open=False):
                    gr.Markdown(
                        "**ConvNeXt-Tiny** — ImageNet on-egitimli, 28M parametre, "
                        "Progressive Unfreezing ile fine-tune.\n\n"
                        "**Custom CNN** — Residual + SE Attention + Multi-Scale, "
                        "~1.3M parametre, sifirdan egitim.\n\n"
                        "**Ensemble** — her iki modelin ciktilarinin agirlikli ortalamasi."
                    )

                with gr.Accordion("Grad-CAM nedir?", open=False):
                    gr.Markdown(
                        "Grad-CAM (Gradient-weighted Class Activation Mapping), "
                        "modelin karar verirken goruntunun hangi bolgelerine baktigini "
                        "gosteren isi haritasidir. **Kirmizi/sari bolgeler** modelin en "
                        "yogun dikkat verdigi alanlardir. Iyi egitilmis bir modelin "
                        "kanama bolgesine odaklanmasi beklenir."
                    )

            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_input = gr.Image(
                            label="CT Goruntusu",
                            type="pil",
                            height=420,
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            predict_btn = gr.Button(
                                "Analiz Et",
                                variant="primary",
                                size="lg",
                                scale=3,
                            )
                            clear_btn = gr.Button(
                                "Temizle",
                                variant="secondary",
                                size="lg",
                                scale=1,
                            )
                        status_display = gr.HTML(_status_html("Hazir. Goruntu yukleyin.", "ready"))

                    with gr.Column(scale=6):
                        result_banner = gr.HTML(
                            "<div style='padding:14px 18px;border-radius:10px;background:#f1f5f9;"
                            "color:#64748b;text-align:center;'>Henuz analiz yapilmadi.</div>"
                        )
                        with gr.Tabs(elem_classes="result-tabs"):
                            with gr.Tab("Olasilik Skorlari"):
                                output_label = gr.Label(
                                    label="Sinif Olasiliklari",
                                    num_top_classes=2,
                                    show_label=False,
                                )
                            with gr.Tab("Grad-CAM"):
                                gradcam_output = gr.Image(
                                    label="Isi haritasi (model dikkat bolgeleri)",
                                    height=380,
                                    show_label=False,
                                )
                            with gr.Tab("Ensemble Detay"):
                                ensemble_display = gr.HTML(
                                    "<div style='padding:14px;color:#64748b;'>"
                                    "Ensemble secildiginde iki modelin ayri skorlari ve "
                                    "birlesim degerleri burada gosterilir.</div>"
                                )
                            with gr.Tab("Detayli Rapor"):
                                detail_output = gr.Textbox(
                                    label="",
                                    lines=16,
                                    interactive=False,
                                    show_label=False,
                                    show_copy_button=True,
                                )

                gr.Markdown("### Hazir Ornekler")
                gr.Examples(
                    examples=_example_paths(),
                    inputs=image_input,
                    label="",
                    examples_per_page=6,
                )

        gr.HTML(
            "<div class='disclaimer'><b>Klinik Uyari.</b> "
            "Bu sistem BM 480 Derin Ogrenme dersi kapsaminda gelistirilmis arastirma amacli "
            "bir prototiptir. Tibbi teshis ya da tedavi kararlari icin kullanilamaz. "
            "Tum klinik kararlar yetkili saglik uzmani tarafindan alinmalidir.</div>"
        )

        gr.HTML(
            "<div class='app-footer'>"
            "<strong>Veri:</strong> felipekitamura/head-ct-hemorrhage (Kaggle) &middot; 200 goruntu (70/15/15 stratified split)"
            " &nbsp;|&nbsp; <strong>Teknikler:</strong> Transfer Learning, Mixup, Label Smoothing, "
            "Cosine Annealing, Progressive Unfreezing, Gradient Clipping, TTA"
            " &nbsp;|&nbsp; <strong>Siniflar:</strong> Normal, Hemorrhage"
            "</div>"
        )

        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_choice, use_tta, show_gradcam],
            outputs=[result_banner, output_label, gradcam_output, detail_output, ensemble_display, status_display],
        )
        clear_btn.click(
            fn=lambda: (None, "<div style='padding:14px 18px;border-radius:10px;background:#f1f5f9;color:#64748b;text-align:center;'>Henuz analiz yapilmadi.</div>", {}, None, "", "", _status_html("Temizlendi. Hazir.", "ready")),
            inputs=[],
            outputs=[image_input, result_banner, output_label, gradcam_output, detail_output, ensemble_display, status_display],
        )

    return demo


def _is_port_available(server_name: str, port: int) -> bool:
    bind_host = "127.0.0.1" if server_name in {"0.0.0.0", ""} else server_name
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def _pick_server_port(server_name: str, preferred_port: int, max_tries: int = 20) -> int:
    for offset in range(max_tries):
        candidate = preferred_port + offset
        if _is_port_available(server_name, candidate):
            return candidate
    raise OSError(
        f"{preferred_port}-{preferred_port + max_tries - 1} araliginda bos port bulunamadi."
    )


def launch_interface():
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    server_port = _pick_server_port(server_name, preferred_port)

    if server_port != preferred_port:
        print(f"[APP] Port {preferred_port} dolu, {server_port} kullaniliyor.")

    demo = create_interface()
    return demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
    )


if __name__ == "__main__":
    launch_interface()
