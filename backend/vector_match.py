"""
vector_match.py — base de matching vectoriel simple
"""

import math
import random


class VectorDB:
    def __init__(self):
        # Base très légère : (nom, lat, lon, vecteur)
        self.db = [
            ("Paris", 48.8566, 2.3522, [0.1, 0.2, 0.3]),
            ("New York", 40.7128, -74.006, [0.2, 0.1, 0.4]),
            ("Tokyo", 35.6895, 139.6917, [0.3, 0.3, 0.3]),
            ("Tikal", 17.222, -89.623, [0.9, 0.8, 0.7]),
            ("Angkor", 13.412, 103.867, [0.85, 0.75, 0.65]),
        ]

    def search(self, query_vec, top_k=3):
        """Recherche des vecteurs les plus proches (distance euclidienne)."""
        results = []
        for name, lat, lon, vec in self.db:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(query_vec, vec)))
            results.append({
                "label": name,
                "lat": lat,
                "lon": lon,
                "distance": dist,
            })
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]


def dummy_encode(image_path: str):
    """Encode l'image en un vecteur factice (longueur 3)."""
    random.seed(hash(image_path) % 1000)
    return [random.random() for _ in range(3)]


# Instance globale exportée
VECTOR_DB = VectorDB()

