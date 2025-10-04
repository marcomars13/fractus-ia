import sys
import json
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

PROFILE_PATH = "mapillary_out/fractus_profile.json"

def test_new_image(img_path, profile_path=PROFILE_PATH, threshold=0.3):
    # Charger le profil fractus
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Charger l'image à tester
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Impossible de lire l'image: {img_path}")
        sys.exit(1)

    img = cv2.resize(img, (profile["img_size"], profile["img_size"]))
    vec = img.flatten() / 255.0

    # Calculer la distance au vecteur moyen
    mean_vec = np.array(profile["visual"]["mean_vector"]).reshape(1, -1)
    dist = cosine_distances(mean_vec, [vec])[0][0]

    print(f"Image testée: {img_path}")
    print(f"Distance au profil moyen: {dist:.4f}")

    if dist > threshold:
        print("⚠️  Image déviante du pattern visuel")
    else:
        print("✅ Image cohérente avec le pattern visuel")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fractus_test_image.py <chemin_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    test_new_image(img_path)

