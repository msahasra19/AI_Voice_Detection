import os
import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import gradio as gr

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app as fastapi_app
from app.utils import load_audio_file
from app.model import detect_voice

# Hugging Face ZeroGPU compatibility: ZeroGPU requires @spaces.GPU on startup
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
        return "No audio provided"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio = load_audio_file(audio_bytes)
    return detect_voice(audio)

# Full-screen glassmorphic dashboard container
EMBEDDED_DASHBOARD_HTML = """
<style>
body, html {
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    width: 100% !important;
    height: 100% !important;
}
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    background: #080b14 !important;
    border: none !important;
}
footer {
    display: none !important;
}
iframe#forensics-dashboard {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 999999 !important;
}
</style>
<iframe id="forensics-dashboard" src="/custom_ui/index.html"></iframe>
"""

with gr.Blocks(title="Voice Anti-Spoofing & Multilingual Forensics AI") as demo:
    gr.HTML(EMBEDDED_DASHBOARD_HTML)

# Mount all FastAPI routes (/predict, /predict-file, /ablation-report, /health, /docs) onto demo.app
demo.app.include_router(fastapi_app.router)
demo.app.mount("/custom_ui", StaticFiles(directory="static", html=True))

# Intercept static assets so /static/styles.css and /static/script.js resolve directly
@demo.app.middleware("http")
async def intercept_static(request, call_next):
    if request.url.path.startswith("/static/"):
        rel = request.url.path[len("/static/"):]
        target_file = Path("static") / rel
        if target_file.exists() and target_file.is_file():
            return FileResponse(target_file)
    return await call_next(request)

if __name__ == "__main__":
    demo.launch()
