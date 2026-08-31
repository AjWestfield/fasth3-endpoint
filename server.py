"""FastH3 inference server for a Hugging Face Inference Endpoint (custom container).

Loads the FastVideo generator once at startup; POST /generate returns an MP4.
Designed for a single H200 replica with scale-to-zero.
"""
import base64
import io
import os
import tempfile
import time

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = os.environ.get("FASTH3_MODEL", "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2")

app = FastAPI()
GEN = None
LOAD_S = None
LOAD_ERR = None


def _load_model():
    global GEN, LOAD_S, LOAD_ERR
    t0 = time.time()
    try:
        model_path = MODEL_ID
        if not MODEL_ID.startswith("/"):
            # Mirror the proven flow exactly: full snapshot to a local dir,
            # then load from the directory (hub-id loading picks a weight
            # layout the H3 mapper cannot handle).
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(MODEL_ID, local_dir="/tmp/FastH3")
        from fastvideo import VideoGenerator
        GEN = VideoGenerator.from_pretrained(
            model_path,
            num_gpus=1,
            dmd_denoising_steps=[999, 749, 500, 250],
        )
        LOAD_S = round(time.time() - t0, 1)
    except Exception as e:  # surfaced via /health
        LOAD_ERR = repr(e)[:500]


@app.on_event("startup")
def start_loading():
    # Load in a thread so the health route answers immediately.
    import threading
    threading.Thread(target=_load_model, daemon=True).start()


class GenRequest(BaseModel):
    prompt: str
    seconds: int = 15  # 5, 10, or 15
    seed: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": GEN is not None,
            "model_load_s": LOAD_S, "load_error": LOAD_ERR}


@app.post("/generate")
def generate(req: GenRequest):
    if GEN is None:
        return {"error": "model still loading" if not LOAD_ERR else LOAD_ERR}
    t0 = time.time()
    frames = {5: 124, 10: 243, 15: 345}.get(req.seconds, 345)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.mp4")
        GEN.generate_video(
            prompt=req.prompt,
            guidance_scale=1.0,
            num_frames=frames,
            seed=req.seed,
            output_path=out,
        )
        data = open(out, "rb").read()
    return {
        "gen_seconds": round(time.time() - t0, 1),
        "video_b64": base64.b64encode(data).decode(),
        "bytes": len(data),
    }
