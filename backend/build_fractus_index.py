import os
import joblib
import numpy as np
from sklearn.neighbors import KDTree

INDEX_CACHE = {}

def build_fractus_index(features, coords, output_file):
    """
    Construit un index KDTree à partir des features et coordonnées.
    ⚡ Sauvegarde maintenant un tuple (features, coords) au lieu du KDTree seul,
    pour permettre une reconstruction complète plus tard.
    """
    print(f"⚡ Construction KDTree avec {features.shape[0]} entrées...")
    tree = KDTree(features)

    # Sauvegarde : tuple (features, coords)
    joblib.dump((features, coords), output_file)
    print(f"✅ Index (features, coords) sauvegardé dans {output_file}")

    return tree, coords


def load_index_once(path):
    """
    Charge un index en mémoire cache.
    - Si le fichier contient directement un KDTree → on l’utilise (legacy).
    - Si le fichier contient un tuple (features, coords) → on reconstruit le KDTree.
    """
    global INDEX_CACHE
    if path in INDEX_CACHE:
        return INDEX_CACHE[path]

    obj = joblib.load(path)

    if isinstance(obj, KDTree):
        print(f"📂 Chargement KDTree direct (legacy) depuis {path}")
        INDEX_CACHE[path] = (obj, None)
        return INDEX_CACHE[path]

    elif isinstance(obj, tuple) and len(obj) == 2:
        features, coords = obj
        print(f"📂 Reconstruction KDTree depuis tuple (features, coords) [{features.shape}]")
        tree = KDTree(features)
        INDEX_CACHE[path] = (tree, coords)
        return INDEX_CACHE[path]

    else:
        raise TypeError(f"❌ Objet chargé depuis {path} n’est pas un KDTree ni un tuple valide (type={type(obj)})")

