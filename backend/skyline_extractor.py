import cv2
import numpy as np
import os

def extract_skyline_signature(image_path: str, resize=(256, 128)) -> np.ndarray:
    """
    Extrait une signature fractale simplifiée basée sur la skyline/horizon.

    Args:
        image_path (str): Chemin de l'image.
        resize (tuple): Taille de redimensionnement (w, h).

    Returns:
        np.ndarray: Profil de skyline normalisé (1D).
    """
    # Charge image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Impossible de lire {image_path}")

    # Redimensionne
    img = cv2.resize(img, resize)

    # Convertit en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection de contours
    edges = cv2.Canny(gray, 100, 200)

    # Profil vertical (on prend le premier pixel blanc du haut)
    h, w = edges.shape
    skyline = np.zeros(w, dtype=np.int32)

    for x in range(w):
        col = edges[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            skyline[x] = ys[0]  # Premier contour rencontré
        else:
            skyline[x] = h  # Pas trouvé → met en bas

    # Normalise entre [0,1]
    skyline_norm = skyline / h
    return skyline_norm

if __name__ == "__main__":
    # 📂 Dossier des images Mapillary
    img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
    files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if not files:
        raise ValueError("❌ Aucun fichier JPG trouvé dans le dossier !")

    # On prend la première image dispo
    test_img = os.path.join(img_dir, files[0])

    # Exécution
    sig = extract_skyline_signature(test_img)
    print(f"✅ Skyline extraite depuis {test_img}")
    print("   → Shape:", sig.shape)
    print("   → Exemple (10 premières valeurs):", sig[:10])

