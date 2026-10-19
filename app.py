import os
import sys
from pathlib import Path
import uvicorn
import gradio as gr

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app
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

# Gradio demo container mounted at /gradio
with gr.Blocks(title="Voice Anti-Spoofing & Multilingual Forensics AI") as demo:
    gr.Markdown("# Voice Anti-Spoofing & Multilingual Forensics API")
    gr.Markdown("Direct interactive dashboard is served at root [/](/). OpenAPI documentation available at [/docs](/docs).")

# Mount Gradio onto the existing FastAPI app so root / directly serves the custom dashboard
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
