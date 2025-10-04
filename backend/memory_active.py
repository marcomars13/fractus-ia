import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KDTree

MEMORY_FILE = Path("results/fractus_memory.pkl")

class FractusMemory:
    def __init__(self):
        self.vectors = []
        self.coords = []
        self.tree = None

    def add(self, vec, lat, lon):
        """Ajoute un vecteur + coordonnées dans la mémoire et reconstruit l'index."""
        self.vectors.append(vec)
        self.coords.append((lat, lon))
        self._rebuild()

    def _rebuild(self):
        if self.vectors:
            self.tree = KDTree(np.array(self.vectors))

    def query(self, vec, k=3):
        """Retourne les k plus proches voisins de la mémoire active."""
        if self.tree is None:
            return []
        dist, idx = self.tree.query([vec], k=min(k, len(self.vectors)))
        return [(self.coords[i], float(d)) for i, d in zip(idx[0], dist[0])]

    def save(self):
        """Sauvegarde la mémoire dans un fichier pickle."""
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump((self.vectors, self.coords), f)

    def load(self):
        """Recharge la mémoire depuis le fichier si dispo."""
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "rb") as f:
                self.vectors, self.coords = pickle.load(f)
            self._rebuild()

