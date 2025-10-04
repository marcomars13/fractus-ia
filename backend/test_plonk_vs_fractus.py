import os
import csv
import random
from datetime import datetime
from plonk.pipe import PlonkPipeline
from PIL import Image
from fractus_skyline import run_fractus_skyline  # ⚡ ton wrapper Fractus

# ================================
# ⚙️ CONFIG
# ================================
MODEL_ID = "nicolas-dufour/PLONK_YFCC"
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
CSV_PATH = "backend/plonk_vs_fractus.csv"
N_IMAGES = 10  # nombre d'images à tester

# ================================
# 🚀 CHARGEMENT PIPELINE PLONK
# ================================
print("🚀 Chargement du modèle Plonk officiel...")
pipeline = PlonkPipeline(model_path=MODEL_ID, device="cpu")
print("✅ Modèle Plonk chargé !")

# ================================
# 🖼️ SÉLECTION D’IMAGES
# ================================
if not os.path.isdir(IMG_DIR):
    raise FileNotFoundError(f"❌ Dossier images introuvable: {IMG_DIR}")

images = sorted(os.listdir(IMG_DIR))
if len(images) < N_IMAGES:
    raise ValueError(f"❌ Pas assez d’images dans {IMG_DIR} (min {N_IMAGES})")

sampled = random.sample(images, N_IMAGES)

# ================================
# 📍 TEST COMPARATIF
# ================================
results = []
for img_name in sampled:
    img_path = os.path.join(IMG_DIR, img_name)
    print(f"\n🖼️ Test image : {img_name}")

    # Plonk
    try:
        img = Image.open(img_path).convert("RGB")
        out = pipeline(img)
        plonk_lat, plonk_lon = float(out[0][0]), float(out[0][1])
        print(f"   📍 Plonk → lat={plonk_lat:.5f}, lon={plonk_lon:.5f}")
    except Exception as e:
        print(f"⚠️ Erreur Plonk pour {img_name}: {e}")
        plonk_lat, plonk_lon = None, None

# Fractus
try:
    fractus_out = run_fractus_skyline(img_path)
    if isinstance(fractus_out, dict) and "lat" in fractus_out and fractus_out["lat"] is not None:
        fractus_lat, fractus_lon = fractus_out["lat"], fractus_out["lon"]
        print(f"   🔮 Fractus → lat={fractus_lat:.5f}, lon={fractus_lon:.5f}")
    else:
        print(f"⚠️ Fractus a échoué pour {img_name}: {fractus_out}")
        fractus_lat, fractus_lon = None, None
except Exception as e:
    print(f"⚠️ Erreur Fractus pour {img_name}: {e}")
    fractus_lat, fractus_lon = None, None


    results.append([datetime.now().isoformat(), img_name,
                    plonk_lat, plonk_lon, fractus_lat, fractus_lon])

# ================================
# 💾 EXPORT CSV
# ================================
header = ["timestamp", "image", "plonk_lat", "plonk_lon", "fractus_lat", "fractus_lon"]
file_exists = os.path.isfile(CSV_PATH)

with open(CSV_PATH, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerows(results)

print(f"\n📂 Résultats sauvegardés dans {CSV_PATH}")

