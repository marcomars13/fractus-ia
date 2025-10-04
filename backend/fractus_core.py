import numpy as np
from sklearn.neighbors import KDTree

# =====================================================
# Extraction vectorielle (placeholder, garde ta logique interne)
# =====================================================
def extract_vector(image_path):
    """
    Convertit une image en vecteur d'embedding.
    ⚠️ Cette fonction est un placeholder : adapte avec ton vrai modèle si nécessaire.
    """
    rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
    return rng.normal(size=(1024,)).astype(np.float32)


# =====================================================
# Fonction fractus_predict corrigée
# =====================================================
def fractus_predict(query_vec, index=None, coords=None, k=1):
    """
    Recherche le voisin le plus proche avec KDTree et renvoie ses coordonnées.
    Args:
        query_vec: vecteur de requête (1D numpy array)
        index: KDTree déjà construit
        coords: tableau numpy (N,2) des coordonnées [lat,lon] alignées aux vecteurs KDTree
        k: nombre de voisins (par défaut 1)
    Returns:
        dict {"lat": float, "lon": float}
    """
    if index is None or coords is None:
        raise ValueError("❌ fractus_predict requiert un KDTree et les coords associées")

    # reshape pour sklearn
    query_vec = np.asarray(query_vec).reshape(1, -1)

    dist, ind = index.query(query_vec, k=k)
    nn_idx = ind[0][0]
    nn_coord = coords[nn_idx]

    return {"lat": float(nn_coord[0]), "lon": float(nn_coord[1])}


# =====================================================
# Build KDTree utilitaire
# =====================================================
def build_fractus_kdtree(features, coords):
    """
    Construit un KDTree sur features et garde les coords alignées.
    """
    index = KDTree(features, metric="euclidean")
    return index, coords

