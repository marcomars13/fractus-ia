import os
import csv
import numpy as np
from datetime import datetime
from PIL import Image
from plonk.pipe import PlonkPipeline
from fractus_core import run_fractus_full_api
from math import radians, sin, cos, sqrt, atan2

# ================================
# 📂 CONFIG
# ================================
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
CSV_PATH = "backend/plonk_fractus_prefilter_multi.csv"
N_IMAGES = 10  # nombre d’images à tester

# ================================
# 🌍 Fonction distance Haversine
# ================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# ================================
# 🚀 Chargement Plonk
# ================================
print("🚀 Chargement du modèle Plonk officiel...")
pipeline = PlonkPipeline(model_path="nicolas-dufour/PLONK_YFCC", device="cpu")
print("✅ Modèle Plonk chargé !")

# ================================
# 🔮 Fonction Fractus pré-filtre
# ================================
def fractus_prefilter(img):
    try:
        fractus_out = run_fractus_full_api(np.array(img))
        score = fractus_out.get("score", 0.0)
        weight = max(0.0, min(1.0, score * 10))  # échelle simple [0,1]
        img_filtered = img.point(lambda p: p * (1 - weight) + 128 * weight)
        return img_filtered, score
    except Exception as e:
        print(f"⚠️ Erreur Fractus pré-filtre: {e}")
        return img, 0.0

# ================================
# 📍 Fonction prédiction Plonk
# ================================
def predict(image):
    out = pipeline(image)
    if isinstance(out, (list, tuple, np.ndarray)):
        return float(out[0][0]), float(out[0][1])
    raise ValueError(f"❌ Résultat inattendu: {out}")

# ================================
# 🔎 Boucle multi-images
# ================================
images = sorted(os.listdir(IMG_DIR))[:N_IMAGES]
results = []

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    print(f"\n🖼️ Test image : {img_name}")
    img = Image.open(img_path).convert("RGB")

    # Plonk brut
    plonk_lat, plonk_lon = predict(img)
    print(f"   📍 Plonk brut → lat={plonk_lat:.5f}, lon={plonk_lon:.5f}")

    # Fractus pré-filtre
    img_filtered, score = fractus_prefilter(img)

    # Plonk filtré
    plonk_lat_filt, plonk_lon_filt = predict(img_filtered)
    print(f"   📍 Plonk + Fractus pré-filtre → lat={plonk_lat_filt:.5f}, lon={plonk_lon_filt:.5f} (score={score:.4f})")

    # Distance entre brut et filtré
    dist = haversine(plonk_lat, plonk_lon, plonk_lat_filt, plonk_lon_filt)
    print(f"   ↔️ Distance brut vs filtré = {dist:.2f} km")

    results.append([
        datetime.now().isoformat(),
        img_name,
        plonk_lat, plonk_lon,
        plonk_lat_filt, plonk_lon_filt,
        score, dist
    ])

# ================================
# 💾 Export CSV
# ================================
header = ["timestamp", "image", "plonk_lat", "plonk_lon",
          "plonk_lat_filt", "plonk_lon_filt", "fractus_score", "dist_km"]

file_exists = os.path.isfile(CSV_PATH)
with open(CSV_PATH, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerows(results)

print(f"\n📂 Résultats sauvegardés dans {CSV_PATH}")

