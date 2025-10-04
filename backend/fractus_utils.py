"""
fractus_utils.py — utilitaires Fractus isolés de FastAPI
"""

from typing import List

def fractus_transform(seq: str, window: int = 32) -> List[float]:
    """
    Transforme une séquence en scores fractals.
    (Copié depuis main.py pour usage sans FastAPI)
    """
    out = []
    for i in range(0, len(seq), window):
        chunk = seq[i:i+window]
        score = sum(ord(c) for c in chunk) % 3  # ⚠️ copie de ta logique réelle
        out.append(score)
    return out

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points GPS (lat1, lon1) et (lat2, lon2) en km.
    """
    R = 6371.0  # rayon de la Terre en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

