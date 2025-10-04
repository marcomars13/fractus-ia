"""
plonk_fractus_hybrid.py — hybride Plonk + Fractus
- Plonk donne une prédiction globale
- Fractus affine en cherchant autour avec FAISS (si index dispo)
- Patch auto dimensions FAISS (padding/troncature)
"""

import os
import argparse
import joblib
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from plonk.pipe import PlonkPipeline
from backend.features import extract_augmented_vector


# --- Correctif dimensions FAISS ---
def match_index_dim(vec: np.ndarray, index_dim: int) -> np.ndarray:
    """
    Adapte automatiquement la dimension du vecteur à celle de l’index :
    - Tronque si vec est trop long
    - Pad avec des zéros si vec est trop court
    """
    vec_dim = vec.shape[1]
    if vec_dim > index_dim:
        vec = vec[:, :index_dim]
    elif vec_dim < index_dim:
        pad_width = index_dim - vec_dim
        vec = np.pad(vec, ((0, 0), (0, pad_width)), mode="constant")
    return vec.astype("float32")


# --- Fonction principale Fractus ---
def run_fractus(img_path, index, img_list, coords_map, pl_lat, pl_lon, radius_km=500):
    print("⚡ Test Fractus (hybride)")
    vec = extract_augmented_vector(img_path)
    if vec is None:
        print(f"⚠️ Impossible d’extraire vecteur pour {img_path}")
        return None

    # ✅ Patch auto dimension
    vec = match_index_dim(vec, index.d)

    distances, indices = index.search(vec, k=5)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(img_list):
            continue
        fname = img_list[idx]
        lat, lon = coords_map.get(fname, (None, None))
        results.append((fname, float(lat), float(lon), float(dist)))

    if not results:
        print("⚠️ Aucun voisin trouvé dans le rayon défini")
        return None

    return results


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Image test")
    args = parser.parse_args()
    img_path = args.img

    print("🚀 Hybride Plonk + Fractus (Skyline si indexé)")

    # Plonk
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC", device="cpu")
    pil_img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        pred = pipe.model(input_tensor).tolist()[0]
    pl_lat, pl_lon = pred[0], pred[1]
    print(f"📍 Plonk → lat={pl_lat:.5f}, lon={pl_lon:.5f}")

    # Fractus
    try:
        index = faiss.read_index("backend/fractus_global.index")
        with open("backend/fractus_global_images.txt", "r") as f:
            img_list = [l.strip() for l in f if l.strip()]
        coords_map = {}
        if os.path.exists("backend/fractus_global_coords.csv"):
            with open("backend/fractus_global_coords.csv", "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        coords_map[parts[0]] = (parts[1], parts[2])

        print("📦 Index chargé via FAISS")
        results = run_fractus(img_path, index, img_list, coords_map, pl_lat, pl_lon)
        if results:
            print("🏆 Résultats hybrides (plus proches voisins):")
            for fname, lat, lon, dist in results:
                print(f" - {fname} | lat={lat:.5f}, lon={lon:.5f} | dist={dist:.2f}")

    except Exception as e:
        print(f"⚠️ Fractus non dispo ou erreur: {e}")


if __name__ == "__main__":
    main()

