"""
constraints.py — pipeline hybride Fractus + Plonk + contraintes
Révision finale avec support explicite du fallback
"""

from dataclasses import dataclass
import os
import random

# Imports internes
from skyline_enhancer import skyline_match
from spectral_analyzer import analyze_spectrum
from vector_match import dummy_encode, VECTOR_DB
from memory_active import MemoryActive
from fallback_fractus import fallback_predict

# ------------------------------
# Feature flags
# ------------------------------
@dataclass
class Flags:
    use_solar: bool = bool(int(os.getenv("USE_SOLAR", "0")))
    use_dem: bool = bool(int(os.getenv("USE_DEM", "0")))
    use_calib: bool = bool(int(os.getenv("USE_CALIB", "0")))
    use_multi: bool = False
    use_skyline: bool = False
    use_vector: bool = False
    use_memory: bool = False
    use_spectral: bool = False
    use_fallback: bool = False


@dataclass
class Weights:
    w_fractus: float = 1.0
    w_solar: float = 0.0
    w_dem: float = 0.0
    w_calib: float = 0.0
    w_skyline: float = 0.0
    w_vector: float = 0.0
    w_memory: float = 0.0
    w_spectral: float = 0.0
    w_fallback: float = 0.0


# ------------------------------
# Dummy prédictions (remplace Plonk)
# ------------------------------
def dummy_predictions():
    return [
        (48.8566, 2.3522, 1.0, 1.0),   # Paris
        (40.7128, -74.006, 1.0, 1.0),  # New York
        (35.6895, 139.6917, 1.0, 1.0), # Tokyo
    ]


# ------------------------------
# Fonction principale
# ------------------------------
def predict_hybrid(image_path: str, flags: Flags):
    preds = dummy_predictions()
    results = []

    for lat, lon, score, weight in preds:
        w = weight

        # Calibration
        if flags.use_calib:
            w *= 1.2

        # Contraintes solaires / DEM
        if flags.use_solar:
            w *= 1.1
        if flags.use_dem:
            w *= 1.1

        # Skyline
        if flags.use_skyline:
            try:
                res = skyline_match(image_path)
                if res.get("label_best") == "ville":
                    w *= 1.1
            except Exception:
                pass

        # Spectral
        if flags.use_spectral:
            try:
                spec = analyze_spectrum(image_path)
                if spec.get("category") == "urbain/périodique":
                    w *= 1.1
                elif spec.get("category") == "forêt/dune":
                    w *= 1.05
            except Exception:
                pass

        # Vector matching
        if flags.use_vector and VECTOR_DB is not None and dummy_encode is not None:
            try:
                qvec = dummy_encode(image_path)
                matches = VECTOR_DB.search(qvec, top_k=1)
                if matches and matches[0]["distance"] < 1.5:
                    w *= 1.2
            except Exception:
                pass

        # Memory active
        if flags.use_memory:
            mem = MemoryActive()
            adjusted = mem.auto_adjust((lat, lon), top_k=3)
            if adjusted:
                lat, lon = adjusted["adjusted_pred"]["lat"], adjusted["adjusted_pred"]["lon"]
                w *= 1.3

        results.append((lat, lon, score, w))

    # ------------------------------
    # Ajout fallback si activé
    # ------------------------------
    if flags.use_fallback:
        try:
            fb = fallback_predict(image_path)
            results.append((fb["lat"], fb["lon"], 1.0, fb["score"]))
        except Exception as e:
            print(f"[WARN] Fallback error: {e}")

    return results

