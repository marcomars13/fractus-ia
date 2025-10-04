from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import importlib, glob, hashlib
from typing import Optional

# Import Fractus et Plonk
from backend.fallback_fractus import fallback_predict as run_fractus
from backend.plonk_model import run_plonk_api

app = FastAPI(title="Fractus + Plonk API", version="1.0")

# ✅ Route santé
@app.get("/health")
def health():
    return {"ok": True}

# ✅ Route principale : Fractus → Plonk
@app.post("/infer/plonk")
async def infer_plonk(file: UploadFile = File(...)):
    try:
        # Sauvegarde temporaire
        contents = await file.read()
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Étape 1 : Fractus
        fractus_out = run_fractus(temp_path)

        # Étape 2 : Plonk (actuellement sans couplage direct)
        plonk_out = run_plonk_api(temp_path)

        return {
            "filename": file.filename,
            "fractus": fractus_out,
            "plonk": plonk_out
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 🔎 Debug utilitaire
def _file_info(path: str) -> dict:
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        return {"path": os.path.abspath(path), "size_mb": round(size_mb, 2), "sha256_16": sha256}
    except Exception as e:
        return {"path": path, "error": str(e)}

def _find_plonk_weights(pm) -> Optional[dict]:
    # Cherche des attributs usuels
    for attr in ["WEIGHTS_PATH", "CHECKPOINT_PATH", "MODEL_PATH", "PLONK_WEIGHTS"]:
        p = getattr(pm, attr, None)
        if isinstance(p, str) and os.path.exists(p):
            return {"from_attr": attr, "file": _file_info(p)}
    # Sinon, cherche dans tout le projet
    candidates = []
    for ext in ("*.pt", "*.pth"):
        candidates += glob.glob(f"**/{ext}", recursive=True)
    plonkish = [c for c in candidates if "plonk" in c.lower()]
    plonkish = sorted(plonkish, key=lambda x: os.path.getsize(x), reverse=True)
    return {"candidates": [_file_info(c) for c in plonkish[:10]]}

# ✅ Route debug Plonk
@app.get("/debug/plonk")
def debug_plonk():
    info = {}
    try:
        pm = importlib.import_module("backend.plonk_model")
        info["module"] = str(pm)

        # Torch infos
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["torch_cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception as e:
            info["torch_info_error"] = str(e)

        # Plonk package version
        try:
            import importlib.metadata as md
            info["plonk_pkg_version"] = md.version("plonk") if "plonk" in md.packages_distributions() else "unknown"
        except Exception:
            info["plonk_pkg_version"] = "unknown"

        # Mode du modèle si dispo
        for attr in ["MODEL", "model"]:
            mdl = getattr(pm, attr, None)
            if mdl is not None:
                info["model_class"] = mdl.__class__.__name__
                try:
                    info["model_training"] = bool(getattr(mdl, "training", False))
                except Exception:
                    pass

        # Check des poids
        info["weights"] = _find_plonk_weights(pm)

        # Variables env
        info["env_device"] = os.environ.get("PLONK_DEVICE", "default")
        info["env_precision"] = os.environ.get("PLONK_PRECISION", "default")

    except Exception as e:
        info["error"] = str(e)
    return JSONResponse(info)

