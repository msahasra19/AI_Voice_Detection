import gradio as gr
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from app.main import app as fastapi_app
from app.utils import load_audio_file
from app.model import detect_voice

def analyze_audio_gradio(audio_path):
    if not audio_path:
        return "Please upload or record an audio file.", 0.0, "N/A", {}, ""
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio = load_audio_file(audio_bytes)
        result = detect_voice(audio)
        verdict = str(result["classification"].value if hasattr(result["classification"], "value") else result["classification"])
        risk = round(float(result["deepfake_risk_score"]), 3)
        lang = f"{result.get('detected_language', 'Unknown')} ({result.get('language_confidence', 0)*100:.1f}%)"
        indicators = {
            "SNR (dB)": round(float(result["acoustic_indicators"].snr_db), 2),
            "Pitch F0 Median (Hz)": round(float(result["acoustic_indicators"].pitch_f0_median), 2),
            "Pitch Stability": round(float(result["acoustic_indicators"].pitch_stability), 3),
            "Silence Ratio": round(float(result["acoustic_indicators"].silence_ratio), 3),
            "Clipping Detected": bool(result["acoustic_indicators"].clipping_detected)
        }
        explanations = "\n".join(f"• {e}" for e in result.get("explainability", []))
        return verdict, risk, lang, indicators, explanations
    except Exception as exc:
        return f"Error: {str(exc)}", 0.0, "Error", {}, str(exc)

# Build Comprehensive Gradio UI
with gr.Blocks(title="Voice Forensics AI") as demo:
    gr.Markdown("# 🎙️ Voice Anti-Spoofing & Multilingual Forensics AI")
    gr.Markdown("Dual-branch hybrid system combining supervised Random Forest ML with physical signal acoustics (SNR, pitch dynamics) and local Faster Whisper.")
    
    with gr.Tab("Interactive Voice Scanner"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload or Record Voice Sample")
                btn_scan = gr.Button("🔍 Analyze Audio Forensics", variant="primary")
            with gr.Column():
                out_verdict = gr.Textbox(label="Classification Verdict")
                out_risk = gr.Number(label="Deepfake Risk Score (0.0 to 1.0)")
                out_lang = gr.Textbox(label="Detected Language & Confidence")
                out_indicators = gr.JSON(label="Physical Acoustic Indicators")
                out_explain = gr.Textbox(label="Forensic Explanations", lines=4)
        btn_scan.click(
            fn=analyze_audio_gradio,
            inputs=[audio_input],
            outputs=[out_verdict, out_risk, out_lang, out_indicators, out_explain]
        )

    with gr.Tab("Custom Glassmorphic Forensics Dashboard"):
        gr.HTML('<iframe src="/static/index.html" style="width:100%; height:900px; border:none; border-radius:12px;"></iframe>')

    with gr.Tab("API Endpoints & Swagger Docs"):
        gr.Markdown("### REST API Documentation\nAccess the full interactive OpenAPI Swagger specification at [`/docs`](/docs).")

# Mount existing FastAPI routes and static files onto demo.app
demo.app.include_router(fastapi_app.router)
static_dir = Path("static")
if static_dir.exists():
    demo.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    demo.launch()
