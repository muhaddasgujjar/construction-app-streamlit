"""
ArchitectXpert — FastAPI backend
Phase 5: Dimension-aware Pix2Pix API
------------------------------------
Endpoints:
 - GET  /health         → health check
 - POST /predict        → generate floorplan given width, depth, px_per_ft
"""

import os
import io
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image, ImageEnhance
import time
import random

# --- internal imports (ensure src package import works) ---
if __package__ in (None, ""):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import UNetGenerator        # noqa: E402
from src.conditioners import make_dimension_map  # noqa: E402

# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(ROOT, "checkpoints_dim", "best_dim.pt")
GENERATED_DIR = os.path.join(ROOT, "api_generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
print(f"[API] Loading model from {CKPT_PATH} on {device}")
G = UNetGenerator(in_channels=4, out_channels=3, base=64).to(device)
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    G.load_state_dict(ckpt.get("G", ckpt), strict=False)
G.eval()
print("[API] Model ready ✓")

app = FastAPI(title="ArchitectXpert API", version="1.0")

def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2

def generate_floorplan(width: float, depth: float, px_per_ft: int = 18):
    """
    Loads a random real A input from CubiCasa pairs if available,
    otherwise fall back to a synthetic edge map. Applies dimension map,
    runs generator, and post-processes result for realism.
    """
    H = W = 256
    PAIR_A = os.path.join(ROOT, "data", "cubicasa_pairs", "A")
    tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    if os.path.isdir(PAIR_A) and len(os.listdir(PAIR_A)) > 0:
        # use a real sample from data/A
        sample_path = os.path.join(PAIR_A, random.choice(os.listdir(PAIR_A)))
        a_img = Image.open(sample_path).convert("RGB")
        a_tensor = tf(a_img).unsqueeze(0)
    else:
        # fallback synthetic edges
        import torch.nn.functional as F
        rand = torch.randn(1, 1, H, W)
        edges = F.max_pool2d(rand, 3, 1, 1) - rand
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)
        a_tensor = edges.repeat(1, 3, 1, 1) * 2 - 1

    # dimension map
    w_tensor = torch.tensor([width], dtype=torch.float32)
    d_tensor = torch.tensor([depth], dtype=torch.float32)
    dim_map = make_dimension_map(1, H, W, w_tensor, d_tensor)

    if isinstance(a_tensor, torch.Tensor):
        x = torch.cat([a_tensor, dim_map], dim=1).to(device)
    else:
        x = torch.cat([a_tensor.to(device), dim_map.to(device)], dim=1)

    with torch.no_grad():
        y = G(x)
    y = denorm(y).cpu().squeeze(0)
    out_img = transforms.ToPILImage()(y)

    out_img = ImageEnhance.Contrast(out_img).enhance(1.6)
    out_img = ImageEnhance.Brightness(out_img).enhance(1.05)
    out_img = ImageEnhance.Sharpness(out_img).enhance(1.2)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"gen_{width:.1f}x{depth:.1f}_{timestamp}.png"
    out_path = os.path.join(GENERATED_DIR, fname)
    out_img.save(out_path)
    return out_path

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

class PredictRequest(BaseModel):
    width: float
    depth: float
    px_per_ft: int = 18

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        out_path = generate_floorplan(req.width, req.depth, req.px_per_ft)
        return FileResponse(out_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return JSONResponse({
        "message": "Welcome to ArchitectXpert API",
        "usage": {"GET /health": "check API status", "POST /predict": "{width, depth, px_per_ft}"}
    })
