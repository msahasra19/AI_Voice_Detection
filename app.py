import os
import gradio as gr
from app.main import app

# Gradio mounts your existing FastAPI app and custom frontend dashboard
demo = gr.Blocks(title="AI Voice Detection")

# Mount Gradio onto the existing FastAPI app
app = gr.mount_gradio_app(app, demo, path="/gradio")

# On Hugging Face Spaces, the container automatically serves `app` on port 7860.
# Only run uvicorn manually when running locally:
if __name__ == "__main__" and "SPACE_ID" not in os.environ:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
