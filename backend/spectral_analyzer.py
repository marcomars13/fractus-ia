"""
spectral_analyzer.py — Analyse spectrale FFT des signatures Fractus
Objectif :
- Extraire la signature fractale (ou vectorielle) d'une image.
- Passer cette signature en transformée de Fourier (FFT).
- Identifier une fréquence dominante → utile pour détecter périodicité
  (forêts, dunes, façades d'immeubles, etc.).

Sortie :
- vecteur spectral
- fréquence dominante
- catégorie heuristique simple
"""

import numpy as np
from typing import Dict, Any

# ------------------------------
# Dummy encodeur fractal (à remplacer par fractus_transform réel)
# ------------------------------
def dummy_signature(image_path: str, dim: int = 128) -> np.ndarray:
    """Retourne une signature pseudo-aléatoire stable basée sur le nom du fichier."""
    seed = sum(ord(c) for c in image_path) % 10007
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)


# ------------------------------
# Analyse spectrale
# ------------------------------
def analyze_spectrum(image_path: str) -> Dict[str, Any]:
    """
    Applique une FFT sur la signature fractale de l'image.
    Retourne fréquence dominante + classification heuristique.
    """
    sig = dummy_signature(image_path)
    fft_vals = np.fft.rfft(sig)  # FFT réelle
    mags = np.abs(fft_vals)

    # Fréquence dominante (hors DC = index 0)
    if len(mags) > 1:
        dom_idx = int(np.argmax(mags[1:]) + 1)
        dom_freq = dom_idx / len(sig)
    else:
        dom_idx = 0
        dom_freq = 0.0

    # Heuristique simple
    if dom_freq < 0.1:
        category = "plaine"
    elif dom_freq < 0.25:
        category = "forêt/dune"
    else:
        category = "urbain/périodique"

    return {
        "ok": True,
        "image": image_path,
        "signature_len": len(sig),
        "dom_index": dom_idx,
        "dom_freq": round(dom_freq, 4),
        "category": category,
    }

