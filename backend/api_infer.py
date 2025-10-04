#!/usr/bin/env python3
"""
api_infer.py — Serveur FastAPI pour Plonk + Fractus
"""

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

# ✅ Imports corrigés
from backend.plonk_model import run_plonk_api
from backend.fractus_core import run_fractus_full_api

app = FastAPI()


# ------------------ Endpoints ------------------

@app.post("/infer/plonk")
async def infer_plonk(file: UploadFile = File(...)):
    """Inference Plonk seule (coordonnées GPS)."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        result = run_plonk_api(img)
        return {"status": "ok", "result": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/infer/fractus")
async def infer_fractus(file: UploadFile = File(...)):
    """Inference Fractus seule (score fractal)."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        result = run_fractus_full_api(img, profile="default")
        return {"status": "ok", "result": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/infer/plonk_fractus")
async def infer_plonk_fractus(file: UploadFile = File(...)):
    """Ancienne version : renvoie uniquement Fractus mais via endpoint mixte."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        result = run_fractus_full_api(img, profile="default")
        return {"status": "ok", "result": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/infer/plonk_plus_fractus")
async def infer_plonk_plus_fractus(file: UploadFile = File(...)):
    """Endpoint combiné : Plonk (GPS) + Fractus (score fractal)."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Étape 1 : localisation Plonk
        plonk_res = run_plonk_api(img)

        # Étape 2 : score fractal
        fractus_res = run_fractus_full_api(img, profile="default")

        # Réponse combinée
        return {
            "status": "ok",
            "plonk": plonk_res,
            "fractus": fractus_res
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

