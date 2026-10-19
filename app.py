import os
import sys
from pathlib import Path
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

# Minimal Gradio demo container required by Hugging Face Gradio SDK
with gr.Blocks(title="Voice Anti-Spoofing & Multilingual Forensics AI") as demo:
    pass

# Mount all FastAPI routes (/predict, /predict-file, /ablation-report, /health, /docs) onto demo.app
demo.app.include_router(fastapi_app.router)

# Intercept root and static asset requests to serve the original custom UI directly
@demo.app.middleware("http")
async def custom_frontend_interceptor(request, call_next):
    path = request.url.path
    if path in ("/", "/index.html"):
        index_file = Path("static") / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    if path.startswith("/static/"):
        rel_path = path[len("/static/"):]
        target_file = Path("static") / rel_path
        if target_file.exists() and target_file.is_file():
            return FileResponse(target_file)
    return await call_next(request)

if __name__ == "__main__":
    demo.launch()
