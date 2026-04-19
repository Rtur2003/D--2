import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")

import gradio as gr

from config import DEVICE, MODELS_DIR, CLASS_NAMES, IMG_SIZE
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model
from gradcam import GradCAM, get_target_layer, overlay_cam_on_image


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:        #F7F6F3;
    --bg-1:      #FFFFFF;
    --bg-2:      #F0EEE9;
    --bg-3:      #E8E5DF;
    --ink:       #111110;
    --ink-2:     #2C2C2A;
    --ink-3:     #5A5956;
    --ink-4:     #8A8884;
    --border:    #E0DDD8;
    --border-2:  #C8C5C0;
    --accent:    #2563EB;
    --accent-bg: #EFF4FF;
    --green:     #16A34A;
    --green-bg:  #F0FDF4;
    --green-text:#15803D;
    --red:       #DC2626;
    --red-bg:    #FEF2F2;
    --red-text:  #B91C1C;
    --radius:    10px;
    --radius-s:  7px;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--ink) !important;
    min-height: 100vh !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }
.gradio-container > .main { padding: 0 !important; }
.gradio-container .gap { gap: 0 !important; }

/* TOPBAR */
.ns-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 54px;
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}
.ns-brand { display: flex; align-items: center; gap: 12px; }
.ns-logo {
    width: 32px; height: 32px;
    background: var(--ink);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.ns-title { font-size: 14px; font-weight: 600; color: var(--ink); letter-spacing: -0.01em; }
.ns-version {
    font-size: 11px; color: var(--ink-4); font-family: var(--mono);
    background: var(--bg-2); border: 1px solid var(--border);
    padding: 2px 8px; border-radius: 4px;
}
.ns-tags { display: flex; gap: 6px; align-items: center; }
.ns-tag {
    font-size: 11px; font-family: var(--mono); color: var(--ink-3);
    background: var(--bg-2); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 5px;
    display: flex; align-items: center; gap: 5px;
}
.ns-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

/* BODY LAYOUT */
.ns-body {
    display: grid !important;
    grid-template-columns: 280px 1fr !important;
    min-height: calc(100vh - 54px);
}

/* SIDEBAR */
.ns-sidebar {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
    display: flex !important;
    flex-direction: column !important;
}
.ns-block { padding: 18px 20px; border-bottom: 1px solid var(--border); }
.ns-block-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-4);
    margin-bottom: 12px; font-family: var(--mono);
}

/* UPLOAD */
.gr-image {
    border: 1.5px dashed var(--border-2) !important;
    border-radius: var(--radius-s) !important;
    background: var(--bg-2) !important;
    transition: all .2s !important;
    min-height: 170px !important;
}
.gr-image:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
}

/* RADIO */
.gr-form, .gr-panel { background: transparent !important; border: none !important; box-shadow: none !important; }
.gr-radio-group { gap: 5px !important; }
.gr-radio-group label {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-s) !important;
    padding: 10px 14px !important;
    background: var(--bg-2) !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
.gr-radio-group label:hover {
    background: var(--bg-1) !important;
    border-color: var(--border-2) !important;
    color: var(--ink) !important;
}
.gr-radio-group label:has(input:checked) {
    background: var(--accent-bg) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 500 !important;
}

/* RUN BUTTON */
#run-btn {
    background: var(--ink) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-s) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 13px 20px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all .18s !important;
    margin: 0 !important;
}
#run-btn:hover { background: var(--ink-2) !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0) !important; }

/* META CARDS */
.ns-meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.ns-meta-card {
    background: var(--bg-2); border: 1px solid var(--border);
    border-radius: var(--radius-s); padding: 9px 11px;
}
.ns-meta-k { font-size: 9px; font-family: var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-4); margin-bottom: 3px; }
.ns-meta-v { font-size: 12px; font-weight: 600; color: var(--ink-2); font-family: var(--mono); }

.ns-disc {
    margin-top: auto; padding: 14px 20px;
    font-size: 10px; font-family: var(--mono); color: var(--ink-4);
    letter-spacing: 0.04em; text-align: center;
    border-top: 1px solid var(--border); line-height: 1.8;
}

/* MAIN */
.ns-main { background: var(--bg) !important; }

/* PANEL */
.ns-panel {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.ns-panel-head {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg-2);
}
.ns-panel-title {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-3); font-family: var(--mono);
}
.ns-panel-badge {
    font-size: 9px; font-family: var(--mono); color: var(--accent);
    background: var(--accent-bg); border: 1px solid #BFDBFE;
    padding: 2px 8px; border-radius: 4px;
}
.ns-panel-body { padding: 16px; }

/* LABEL */
.gr-label {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
.gr-label .label-container,
.gr-label .label-container > div { display: block !important; visibility: visible !important; }
.gr-label .label-container .bar-container { margin-bottom: 10px !important; }
.gr-label .label-container .bar { height: 6px !important; border-radius: 3px !important; background: var(--accent) !important; }
.gr-label .label-container .bar-bg { height: 6px !important; border-radius: 3px !important; background: var(--bg-3) !important; }
.gr-label .label-container .category { font-size: 13px !important; font-weight: 500 !important; color: var(--ink) !important; font-family: var(--sans) !important; }
.gr-label .label-container .confidence { font-size: 13px !important; font-weight: 600 !important; color: var(--ink-2) !important; font-family: var(--mono) !important; }

/* GRADCAM */
.gr-image-output {
    border: none !important;
    border-radius: 0 !important;
    overflow: hidden !important;
    background: var(--bg-3) !important;
}

/* REPORT HTML */
.ns-report-wrap { padding: 16px 24px 24px; }
.ns-report-panel {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.ns-report-head {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg-2);
}
.ns-report-body { padding: 20px 22px; }
.ns-report-divider {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 18px;
}
.ns-report-divider-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ns-report-divider-label {
    font-size: 11px; font-weight: 500; color: var(--ink-4);
    font-family: var(--mono); letter-spacing: 0.06em; text-transform: uppercase;
}
.ns-report-divider-line { flex: 1; height: 1px; background: var(--border); }
.ns-stat-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
    margin-bottom: 20px;
}
.ns-stat-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-s);
    padding: 11px 13px;
}
.ns-stat-label {
    font-size: 9px; font-family: var(--mono); letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-4); margin-bottom: 5px;
}
.ns-stat-value { font-size: 15px; font-weight: 600; color: var(--ink); }
.ns-stat-value.positive { color: var(--green-text); }
.ns-stat-value.negative { color: var(--red-text); }
.ns-stat-value.mono { font-family: var(--mono); font-size: 12px; }
.ns-bar-row { margin-bottom: 13px; }
.ns-bar-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
}
.ns-bar-name { font-size: 13px; font-weight: 500; color: var(--ink); }
.ns-bar-pct { font-size: 12px; font-weight: 600; font-family: var(--mono); }
.ns-bar-track {
    height: 6px; background: var(--bg-3); border-radius: 3px;
    overflow: hidden; border: 1px solid var(--border);
}
.ns-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.ns-placeholder {
    text-align: center; padding: 32px 20px;
    color: var(--ink-4); font-size: 12px;
    font-family: var(--mono); letter-spacing: 0.04em;
}
.ns-section-sub {
    font-size: 9px; font-family: var(--mono); letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-4);
    margin: 14px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 2px; }

/* GRADIO BLOCK OVERRIDES */
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
"""

_cache = {}


def _load_one_custom(path):
    m = get_custom_cnn()
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval().to(DEVICE)


def _load_models():
    if "convnext" in _cache:
        return
    with open(str(MODELS_DIR / "train_stats.json"), "r") as f:
        stats = json.load(f)

    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(
        str(MODELS_DIR / "convnext_tiny_best.pth"),
        map_location=DEVICE, weights_only=False
    )
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)

    az_path  = MODELS_DIR / "custom_cnn_best.azveri.pth"
    cok_path = MODELS_DIR / "custom_cnn_best.cok.veri.pth"
    old_path = MODELS_DIR / "custom_cnn_best.pth"

    custom_az  = _load_one_custom(az_path)  if az_path.exists()  else \
                 _load_one_custom(old_path) if old_path.exists() else None
    custom_cok = _load_one_custom(cok_path) if cok_path.exists() else None

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])

    _cache.update({
        "convnext":   convnext,
        "custom_az":  custom_az,
        "custom_cok": custom_cok,
        "custom":     custom_az,   # geriye dönük uyumluluk
        "transform":  transform,
    })


def _render_bar(name: str, score: float, is_primary: bool) -> str:
    pct = score * 100
    bar_color = "#16A34A" if is_primary else "#C8C5C0"
    pct_color = "#15803D" if is_primary else "#8A8884"
    return f"""
    <div class="ns-bar-row">
      <div class="ns-bar-top">
        <span class="ns-bar-name">{name}</span>
        <span class="ns-bar-pct" style="color:{pct_color}">{pct:.1f}%</span>
      </div>
      <div class="ns-bar-track">
        <div class="ns-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
      </div>
    </div>"""


def _build_report_html(scores: dict, pred: str, conf: float, model_name: str) -> str:
    diag_color = "positive" if pred.lower() == "normal" else "negative"
    diag_dot_color = "#16A34A" if pred.lower() == "normal" else "#DC2626"

    bars_html = "".join(
        _render_bar(cls, score, cls == pred)
        for cls, score in scores.items()
    )

    return f"""
    <div class="ns-report-wrap">
      <div class="ns-report-panel">
        <div class="ns-report-head">
          <span class="ns-panel-title">Detayli Rapor</span>
          <span class="ns-panel-badge">Analysis Output</span>
        </div>
        <div class="ns-report-body">
          <div class="ns-report-divider">
            <div class="ns-report-divider-dot" style="background:{diag_dot_color}"></div>
            <span class="ns-report-divider-label">Analysis Report</span>
            <div class="ns-report-divider-line"></div>
          </div>
          <div class="ns-stat-grid">
            <div class="ns-stat-card">
              <div class="ns-stat-label">Diagnosis</div>
              <div class="ns-stat-value {diag_color}">{pred}</div>
            </div>
            <div class="ns-stat-card">
              <div class="ns-stat-label">Confidence</div>
              <div class="ns-stat-value">{conf*100:.1f}%</div>
            </div>
            <div class="ns-stat-card">
              <div class="ns-stat-label">Model</div>
              <div class="ns-stat-value mono">{model_name}</div>
            </div>
          </div>
          <div class="ns-section-sub">Class Scores</div>
          {bars_html}
        </div>
      </div>
    </div>"""


def _placeholder_html() -> str:
    return """
    <div class="ns-report-wrap">
      <div class="ns-report-panel">
        <div class="ns-report-head">
          <span class="ns-panel-title">Detayli Rapor</span>
          <span class="ns-panel-badge">Analysis Output</span>
        </div>
        <div class="ns-placeholder">
          // Goruntu yukleyip analizi baslatın
        </div>
      </div>
    </div>"""


def predict(image, model_choice: str):
    if image is None:
        return {}, None, _placeholder_html()

    _load_models()
    tf = _cache["transform"]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    tensor = tf(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p_cnx = torch.softmax(_cache["convnext"](tensor), dim=1)[0].cpu().numpy()
        _az  = _cache["custom_az"]
        _cok = _cache["custom_cok"]
        p_az  = torch.softmax(_az(tensor),  dim=1)[0].cpu().numpy() if _az  else p_cnx
        p_cok = torch.softmax(_cok(tensor), dim=1)[0].cpu().numpy() if _cok else p_cnx

    if "Ensemble" in model_choice:
        probs = 0.5 * p_cnx + 0.5 * p_az
        model_label = "Ensemble"
        cam_model = _cache["convnext"]
        cam_name  = "convnext"
    elif "ConvNeXt" in model_choice:
        probs = p_cnx
        model_label = "ConvNeXt-Tiny"
        cam_model = _cache["convnext"]
        cam_name  = "convnext"
    elif "Cok Veri" in model_choice or "cok" in model_choice.lower():
        probs = p_cok
        model_label = "Custom CNN (Cok Veri)"
        cam_model = _cok
        cam_name  = "custom"
    else:
        probs = p_az
        model_label = "Custom CNN (Az Veri)"
        cam_model = _az
        cam_name  = "custom"

    scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    pred = CLASS_NAMES[probs.argmax()]
    conf = float(probs.max())

    report_html = _build_report_html(scores, pred, conf, model_label)

    try:
        layer = get_target_layer(cam_model, cam_name)
        gcam = GradCAM(cam_model, layer)
        orig = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        cam_map, _, _ = gcam.generate(tf(image).unsqueeze(0))
        overlay = overlay_cam_on_image(orig, cam_map, alpha=0.45)
        gradcam_img = (overlay * 255).astype(np.uint8)
    except Exception as e:
        gradcam_img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        report_html += (
            f'<div style="padding:8px 22px;font-size:11px;'
            f'font-family:var(--mono);color:#DC2626">'
            f'[Grad-CAM error: {e}]</div>'
        )

    return scores, gradcam_img, report_html


def build_ui():
    with gr.Blocks(title="NeuroScan AI") as demo:

        gr.HTML("""
        <div class="ns-topbar">
          <div class="ns-brand">
            <div class="ns-logo">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"
                   stroke="#F7F6F3" stroke-width="1.5">
                <circle cx="8" cy="8" r="5.5"/>
                <circle cx="8" cy="8" r="1.5" fill="#F7F6F3" stroke="none"/>
                <line x1="8" y1="1" x2="8" y2="3.5"/>
                <line x1="8" y1="12.5" x2="8" y2="15"/>
                <line x1="1" y1="8" x2="3.5" y2="8"/>
                <line x1="12.5" y1="8" x2="15" y2="8"/>
              </svg>
            </div>
            <span class="ns-title">NeuroScan AI</span>
            <span class="ns-version">v1.0</span>
          </div>
          <div class="ns-tags">
            <div class="ns-tag"><div class="ns-dot"></div>Modeller hazir</div>
            <div class="ns-tag">PyTorch</div>
            <div class="ns-tag">Grad-CAM</div>
          </div>
        </div>
        """)

        with gr.Row(elem_classes=["ns-body"]):

            with gr.Column(scale=0, min_width=280, elem_classes=["ns-sidebar"]):

                gr.HTML('<div class="ns-block"><div class="ns-block-label">CT Goruntusu</div>')
                image_input = gr.Image(
                    label="", type="pil", height=170,
                    show_label=False, sources=["upload", "clipboard"],
                )
                gr.HTML("</div>")

                gr.HTML('<div class="ns-block"><div class="ns-block-label">Model Secimi</div>')
                model_choice = gr.Radio(
                    choices=[
                        "ConvNeXt-Tiny — Transfer Learning",
                        "Custom CNN Az Veri — 200 Goruntu",
                        "Custom CNN Cok Veri — Buyuk Dataset",
                        "Ensemble — Soft Voting",
                    ],
                    value="Ensemble — Soft Voting",
                    label="", show_label=False,
                )
                gr.HTML("</div>")

                gr.HTML('<div class="ns-block" style="border-bottom:none">')
                predict_btn = gr.Button(
                    "Analizi Baslat", variant="primary", elem_id="run-btn",
                )
                gr.HTML("</div>")

                gr.HTML("""
                <div class="ns-block">
                  <div class="ns-block-label">Sistem</div>
                  <div class="ns-meta-grid">
                    <div class="ns-meta-card">
                      <div class="ns-meta-k">Device</div>
                      <div class="ns-meta-v">CPU</div>
                    </div>
                    <div class="ns-meta-card">
                      <div class="ns-meta-k">Img Size</div>
                      <div class="ns-meta-v">224x224</div>
                    </div>
                    <div class="ns-meta-card">
                      <div class="ns-meta-k">Classes</div>
                      <div class="ns-meta-v">2</div>
                    </div>
                    <div class="ns-meta-card">
                      <div class="ns-meta-k">Framework</div>
                      <div class="ns-meta-v">PyTorch</div>
                    </div>
                  </div>
                </div>
                """)

                gr.HTML("""
                <div class="ns-disc">
                  BM 480 Derin Ogrenme · Proje 2<br>
                  Yalnizca arastirma amaclidir<br>
                  Klinik karar vermek icin kullanilmaz
                </div>
                """)

            with gr.Column(scale=1, elem_classes=["ns-main"]):

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding:24px 24px 0">
                          <div class="ns-panel">
                            <div class="ns-panel-head">
                              <span class="ns-panel-title">Tahmin Skorlari</span>
                              <span class="ns-panel-badge">Softmax</span>
                            </div>
                            <div class="ns-panel-body">
                        """)
                        output_label = gr.Label(
                            label="", num_top_classes=2, show_label=False,
                        )
                        gr.HTML("</div></div></div>")

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding:24px 24px 0">
                          <div class="ns-panel">
                            <div class="ns-panel-head">
                              <span class="ns-panel-title">Grad-CAM Aktivasyon</span>
                              <span class="ns-panel-badge">Harita</span>
                            </div>
                        """)
                        gradcam_output = gr.Image(
                            label="", height=220, show_label=False,
                        )
                        gr.HTML("</div></div>")

                detail_output = gr.HTML(value=_placeholder_html())

        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_choice],
            outputs=[output_label, gradcam_output, detail_output],
        )

    return demo


_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


def launch_interface():
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=_THEME,
        css=CSS,
    )


if __name__ == "__main__":
    launch_interface()
