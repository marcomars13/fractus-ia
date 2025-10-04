import cv2
import numpy as np

def extract_skyline(image_path, vector_size=128):
    """
    Extrait une signature "skyline" d'une image :
    - Convertit en niveaux de gris
    - Détecte les contours avec Canny
    - Calcule un profil vertical (somme des pixels par colonne)
    - Normalise et réduit à un vecteur de taille fixe
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Impossible de lire l'image: {image_path}")

    # Détection de contours (skyline)
    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    # Profil vertical (somme des colonnes)
    profile = np.sum(edges, axis=0)

    # Normalisation
    if np.max(profile) > 0:
        profile = profile / np.max(profile)

    # Redimensionner à taille fixe (vector_size)
    vec = cv2.resize(profile.reshape(1, -1), (vector_size, 1)).flatten()

    # Sécurité : conversion float32
    vec = vec.astype(np.float32)

    return vec

