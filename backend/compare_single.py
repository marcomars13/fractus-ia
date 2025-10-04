import os
import numpy as np
from PIL import Image
import torch
from plonk import PlonkPipeline
from sklearn.metrics.pairwise import haversine_distances

# --- Config ---
IMG_PATH = "/Users/marco/Desktop/Unknown-2.jpeg"   # 🖼️ Photo du Garlaban
GT_LAT, GT_LON = 43.296, 5.528                    # 📍 Coordonnées vraies du Garlaban
INDEX_PATH = "backend/fractus_global.index"       # 📦 Index Fractus
IMAGES_LIST = "backend/fractus_global_images.txt" # 📂 Liste d'images indexées


# --- Utils ---
def km_distance(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 6371 * haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1]


# --- Plonk ---
def run_plonk(img_path):
    print("🚀 Test Plonk")
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC", device="cpu")

    pil_img = Image.open(img_path).convert("RGB")
    pred = pipe([pil_img], batch_size=1)

    lat, lon = float(pred[0][0]), float(pred[0][1])
    dist = km_distance(lat, lon, GT_LAT, GT_LON)
    print(f"📍 Plonk → lat={lat:.5f}, lon={lon:.5f} | erreur={dist:.2f} km")
    return lat, lon, dist


# --- Fractus ---
def run_fractus(img_path):
    print("⚡ Test Fractus")
    import faiss

    # Charger index
    index = faiss.read_index(INDEX_PATH)
    with open(IMAGES_LIST, "r") as f:
        img_list = [l.strip() for l in f.readlines()]

    from fractus_core import extract_vector
    vec = extract_vector(img_path)
    if vec is None:
        print("❌ Impossible d’encoder l’image avec Fractus")
        return None, None, None

    D, I = index.search(vec, k=1)
    best_idx = I[0][0]
    best_img = img_list[best_idx]

    # Ici → on n’a pas de lat/lon dans l’index, donc affichage brut
    print(f"⚡ Plus proche voisin: {best_img} (distance={D[0][0]:.2f})")
    return None, None, None  # si tu veux ajouter des coords, il faut les stocker dans ton index


def main():
    print(f"🖼️ Image test: {IMG_PATH}")
    print(f"🎯 Ground truth: lat={GT_LAT}, lon={GT_LON}")

    # Plonk
    run_plonk(IMG_PATH)

    # Fractus
    run_fractus(IMG_PATH)


if __name__ == "__main__":
    main()

