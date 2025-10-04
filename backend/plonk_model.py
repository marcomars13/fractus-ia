#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper Plonk — adapté à ta version (pipe.py).
"""

import numpy as np
from plonk_official.plonk.pipe import PlonkPipeline

# Chemin correct vers ton modèle
model_path = "/Users/marco/Projets/fractus-ia/plonk_official/models/PLONK_YFCC"
print(f"🚀 Initialisation PlonkPipeline avec {model_path}")
_pipe = PlonkPipeline(model_path, device="cpu")

def run_plonk_api(image_path: str):
    """
    Lance une prédiction Plonk complète (renvoie tuple).
    """
    return _pipe.predict(image_path, return_features=True)

def get_image_embedding(image_path: str) -> np.ndarray:
    """
    Retourne uniquement l'embedding d'une image (vector 1024D).
    """
    out = _pipe.predict(image_path, return_features=True)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        emb = out[1]
        if emb is None:
            raise RuntimeError(f"❌ Aucun embedding retourné pour {image_path}")
        return np.asarray(emb, dtype="float32").reshape(-1)  # flatten (1024,)
    raise RuntimeError(f"❌ Format inattendu pour la sortie Plonk : {type(out)}")

