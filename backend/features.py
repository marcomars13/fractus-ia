"""
features.py — extraction de vecteurs augmentés pour Fractus
"""

import numpy as np
from PIL import Image

from .skyline import compute_skyline_from_path  # ✅ relatif
from .fractus_core import run_fractus_full_api  # ✅ relatif

def extract_augmented_vector(img_path: str) -> np.ndarray:
    """
    Extrait un vecteur augmentée pour une image :
    - Encodeur Fractus
    - + Skyline si dispo
    """
    try:
        pil_img = Image.open(img_path).convert("RGB")
        result = run_fractus_full_api(np.array(pil_img))
        base_vec = result.get("vector", None)
        if base_vec is None:
            print(f"⚠️ Impossible d’extraire vecteur pour {img_path}")
            return None

        vec = np.array(base_vec, dtype="float32").flatten()

        # Ajout skyline si dispo
        try:
            sky_vec = compute_skyline_from_path(img_path)
            if sky_vec is not None:
                vec = np.concatenate([vec, sky_vec])
        except Exception:
            pass

        return vec.reshape(1, -1)

    except Exception as e:
        print(f"⚠️ extract_augmented_vector erreur sur {img_path}: {e}")
        return None

