import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

# =============================
# CONFIG
# =============================
CSV_PATH = "mapillary_out/images.csv"
THUMBS_DIR = "mapillary_out/thumbs"
PROFILE_PATH = "mapillary_out/fractus_profile.json"

IMG_SIZE = 32  # taille de réduction (32x32 → 1024 valeurs)

# =============================
# STEP 1 : PROFIL GÉOGRAPHIQUE
# =============================
def build_geo_profile(df):
    geo = {
        "lat_mean": float(df["lat"].mean()),
        "lat_std": float(df["lat"].std()),
        "lon_mean": float(df["lon"].mean()),
        "lon_std": float(df["lon"].std()),
    }
    return geo

# =============================
# STEP 2 : PROFIL VISUEL
# =============================
def build_visual_profile(thumbs_dir):
    vectors = []
    for path in glob.glob(os.path.join(thumbs_dir, "*.jpg")):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        vec = img.flatten() / 255.0
        vectors.append(vec)
    if not vectors:
        raise RuntimeError("Aucune image trouvée dans thumbs/")
    vectors = np.array(vectors)
    profile = {
        "mean_vector": vectors.mean(axis=0).tolist(),
        "std_vector": vectors.std(axis=0).tolist(),
    }
    return profile

# =============================
# STEP 3 : SAUVEGARDE PROFIL
# =============================
def save_profile(geo_profile, visual_profile, out_path):
    profile = {
        "geo": geo_profile,
        "visual": visual_profile,
        "img_size": IMG_SIZE,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"[OK] Profil fractus sauvegardé → {out_path}")

# =============================
# STEP 4 : TESTER UNE IMAGE
# =============================
def test_new_image(img_path, profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Charger l'image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Impossible de lire {img_path}")
    img = cv2.resize(img, (profile["img_size"], profile["img_size"]))
    vec = img.flatten() / 255.0

    # Comparaison visuelle
    mean_vec = np.array(profile["visual"]["mean_vector"]).reshape(1, -1)
    dist = cosine_distances(mean_vec, [vec])[0][0]

    print(f"Image testée: {img_path}")
    print(f"Distance au profil moyen: {dist:.4f}")
    if dist > 0.3:  # seuil arbitraire
        print("⚠️  Image déviante du pattern visuel")
    else:
        print("✅ Image cohérente avec le pattern visuel")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # Étape 1 : profil géographique
    geo_profile = build_geo_profile(df)
    print("[INFO] Profil géographique:", geo_profile)

    # Étape 2 : profil visuel
    visual_profile = build_visual_profile(THUMBS_DIR)
    print("[INFO] Profil visuel calculé (vecteur moyen et écart-type)")

    # Étape 3 : sauvegarde
    save_profile(geo_profile, visual_profile, PROFILE_PATH)

    # Étape 4 : test (tu peux changer l’image ici)
    sample_img = glob.glob(os.path.join(THUMBS_DIR, "*.jpg"))[0]
    test_new_image(sample_img, PROFILE_PATH)

