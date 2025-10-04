#!/usr/bin/env python3
"""
Fractus Ultimate Model
Fusion de :
 - fractus_parallel (moteur multi-résolution fractal)
 - fractus_model_full (skyline, vector_match, memory)

Usage :
    from fractus_model_ultimate import run_fractus_ultimate_api
"""

import os
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List

# ============================
# ⛓️ Import des modules avancés
# ============================
import skyline_enhancer
try:
    import memory
except ImportError:
    memory = None


# ============================
# ⚙️ Moteur fractal multi-résolution
# ============================
def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    h, w = int(gray.shape[0] * scale), int(gray.shape[1] * scale)
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

def _fft_polar_bins(gray: np.ndarray, rbins: int = 64, tbins: int = 64) -> np.ndarray:
    """FFT + histogramme polaire"""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    t = (np.arctan2(y - cy, x - cx) + np.pi) * tbins / (2 * np.pi)
    t = t.astype(int)
    r = np.clip(r, 0, rbins - 1)
    t = np.clip(t, 0, tbins - 1)
    hist = np.zeros((rbins, tbins), dtype=np.float32)
    for i in range(rbins):
        for j in range(tbins):
            hist[i, j] = mag[(r == i) & (t == j)].sum()
    return hist.flatten()

def compute_fractus_scores(gray: np.ndarray, multi: bool = True) -> np.ndarray:
    """Extrait les features fractales mono ou multi-résolution."""
    if not multi:
        return _fft_polar_bins(gray)

    feats = []
    for scale in [1.0, 0.5, 0.25]:
        gray_scaled = _resize_gray(gray, scale)
        feats.append(_fft_polar_bins(gray_scaled))
    return np.concatenate(feats)


# ============================
# 🚀 API principale
# ============================
def run_fractus_ultimate_api(img_input, profile: str = "default", multi: bool = True):
    """
    Lance Fractus Ultimate (multi-résolution + skyline + vector_match + memory).
    - img_input : chemin d'image ou ndarray RGB
    - profile   : profil fractal
    - multi     : active multi-résolution
    """

    try:
        # 1. Charger image
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("L")
            gray = np.array(img, dtype=np.float32)
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 3:
                gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = img_input.astype(np.float32)
        else:
            raise ValueError("Format image non supporté")

        # 2. Features fractales multi-résolution
        features = compute_fractus_scores(gray, multi=multi)

        # 3. Skyline enhancement
        features = skyline_enhancer.enhance_skyline(features)

        # 4. Vector match
        if memory and hasattr(memory, "vector_match"):
            features = memory.vector_match(features)

        # 5. Score fractal simple
        score = float(np.linalg.norm(features) / (len(features) + 1e-9))

        # 6. Projection simplifiée → coords
        lat = float((features.mean() % 180) - 90)
        lon = float((features.sum() % 360) - 180)

        return {
            "lat": lat,
            "lon": lon,
            "score": score,
            "profile": profile,
            "note": "Fractus Ultimate (multi+skyline+vector+memory)"
        }

    except Exception as e:
        return {
            "lat": 0.0,
            "lon": 0.0,
            "score": 0.0,
            "profile": profile,
            "note": f"Erreur: {e}"
        }


# ============================
# 🔬 Test rapide
# ============================
if __name__ == "__main__":
    print("🚀 Test Fractus Ultimate")
    fake = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    out = run_fractus_ultimate_api(fake)
    print("Résultat:", out)

