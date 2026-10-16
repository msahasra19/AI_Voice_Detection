import gradio as gr
from app.main import app

# Gradio mounts your existing FastAPI app and custom frontend dashboard
demo = gr.Blocks(title="AI Voice Detection")

# Mount Gradio onto the existing FastAPI app
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
