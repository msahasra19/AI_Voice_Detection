import os
import sys
import re
from pathlib import Path
import gradio as gr

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app as fastapi_app
from app.utils import load_audio_file
from app.model import detect_voice

# Hugging Face ZeroGPU compatibility: ZeroGPU requires an event-bound @spaces.GPU function during startup
try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, Exception):
    def gpu_decorator(fn):
        return fn

@gpu_decorator
def predict_voice_sample(audio_path):
    """
    ZeroGPU registered worker for Hugging Face hardware allocation.
    """
    if not audio_path:
        return {}
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio = load_audio_file(audio_bytes)
    result = detect_voice(audio)
    return {
        "classification": str(result.get("classification")),
        "deepfake_risk_score": float(result.get("deepfake_risk_score", 0.0))
    }

# Read custom frontend assets for direct native injection (no sandboxed iframes)
static_dir = BASE_DIR / "static"
index_html = (static_dir / "index.html").read_text(encoding="utf-8")
css_content = (static_dir / "styles.css").read_text(encoding="utf-8")
js_content = (static_dir / "script.js").read_text(encoding="utf-8")

# Extract the body markup between <body> and the closing <script>
body_match = re.search(r"<body>(.*?)<script", index_html, re.DOTALL)
body_content = body_match.group(1) if body_match else index_html

# Custom CSS to blend Gradio container with original dark glassmorphism
combined_css = """
footer { display: none !important; }
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #080b14 !important;
    border: none !important;
}
.container {
    padding-top: 48px !important;
}
""" + css_content

head_tags = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
"""

with gr.Blocks(title="Voice Anti-Spoofing & Multilingual Forensics AI") as demo:
    gr.HTML(body_content)
    dummy_in = gr.Audio(visible=False)
    dummy_out = gr.JSON(visible=False)
    dummy_btn = gr.Button("Analyze", visible=False)
    dummy_btn.click(fn=predict_voice_sample, inputs=[dummy_in], outputs=[dummy_out])

# Mount all FastAPI routes (/predict, /predict-file, /ablation-report, /health, /docs)
demo.app.include_router(fastapi_app.router)

if __name__ == "__main__":
    demo.launch(head=head_tags, css=combined_css, js=js_content)
