"""
fallback_fractus.py — Mode Fallback (Fractus pur + contraintes physiques)
Objectif :
- Fournir une estimation même si Plonk est indisponible.
- Utilise signatures Fractus + filtres physiques simples.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import random


# ------------------------------
# Dummy encodeur fractal (à remplacer par fractus_transform réel)
# ------------------------------
def dummy_fractus_signature(image_path: str, dim: int = 64) -> np.ndarray:
    """Retourne une signature pseudo-aléatoire stable basée sur le nom de fichier."""
    seed = sum(ord(c) for c in image_path) % 10007
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)


# ------------------------------
# Filtres physiques simplifiés
# ------------------------------
def apply_physical_constraints(lat: float, lon: float, flags: Dict[str, bool]) -> Tuple[float, float]:
    """
    Applique des ajustements factices selon les contraintes activées.
    - solar : décale légèrement en latitude
    - dem   : décale légèrement en altitude/longitude
    - biome : bruit contrôlé
    """
    if flags.get("solar", False):
        lat += 0.05  # exemple : contrainte solaire → décale un peu
    if flags.get("dem", False):
        lon += 0.05  # exemple : contrainte altitude
    if flags.get("biome", False):
        lat += random.uniform(-0.02, 0.02)
        lon += random.uniform(-0.02, 0.02)
    return lat, lon


# ------------------------------
# Fallback principal
# ------------------------------
def fallback_predict(image_path: str, flags: Dict[str, bool] = None) -> Dict[str, Any]:
    """
    Prédit un emplacement basé uniquement sur Fractus + contraintes physiques.
    Retourne {lat, lon, score, source="fallback"}.
    """
    if flags is None:
        flags = {"solar": False, "dem": False, "biome": False}

    # Dummy signature → moyenne pour "générer" des coordonnées
    sig = dummy_fractus_signature(image_path)
    lat = float(sig.mean() * 90.0)   # latitude approx [0,90]
    lon = float(sig.std() * 180.0)   # longitude approx [0,180]

    # Centrer dans des bornes réalistes
    lat = max(-90.0, min(90.0, lat))
    lon = max(-180.0, min(180.0, lon))

    # Appliquer contraintes
    lat, lon = apply_physical_constraints(lat, lon, flags)

    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "score": 0.8,
        "source": "fallback",
    }

