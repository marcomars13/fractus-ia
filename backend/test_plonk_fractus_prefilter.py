import os
import csv
from datetime import datetime
from PIL import Image
import numpy as np

from plonk.pipe import PlonkPipeline
from fractus_core import run_fractus_full_api

IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
CSV_PATH = "backend/plonk_fractus_prefilter.csv"

print("🚀 Chargement du modèle Plonk...")
pipeline = PlonkPipeline(model_path="nicolas-dufour/PLONK_YFCC", device="cpu")
print("✅ Modèle Plonk chargé !")

images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))]
if not images:
    raise FileNotFoundError(f"❌ Aucun fichier trouvé dans {IMG_DIR}")

test_img = os.path.join(IMG_DIR, images[0])
print(f"🖼️ Test sur {test_img}")
img = Image.open(test_img).convert("RGB")

print("🔎 Passage par Fractus pré-filtre...")
try:
    fractus_out = run_fractus_full_api(np.array(img))
    print(f"   ↳ Retour brut Fractus: {fractus_out}")

    score = fractus_out.get("score", 0.0)
    weight = max(0.0, min(1.0, score * 10))  # pondération simple [0-1]

    img_filtered = img.point(lambda p: p * (1 - weight) + 128 * weight)
except Exception as e:
    print(f"⚠️ Erreur Fractus: {e}")
    img_filtered = img

def predict_and_log(image, label):
    out = pipeline(image)
    if isinstance(out, (list, tuple, np.ndarray)):
        lat, lon = float(out[0][0]), float(out[0][1])
    else:
        raise ValueError(f"❌ Résultat inattendu {out}")
    print(f"   📍 {label} → lat={lat:.5f}, lon={lon:.5f}")
    return lat, lon

plonk_lat_raw, plonk_lon_raw = predict_and_log(img, "Plonk brut")
plonk_lat_filt, plonk_lon_filt = predict_and_log(img_filtered, "Plonk + Fractus préfiltre")

header = ["timestamp", "image", "plonk_lat_raw", "plonk_lon_raw", "plonk_lat_filt", "plonk_lon_filt"]
row = [datetime.now().isoformat(), os.path.basename(test_img),
       plonk_lat_raw, plonk_lon_raw, plonk_lat_filt, plonk_lon_filt]

file_exists = os.path.isfile(CSV_PATH)
with open(CSV_PATH, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(row)

print(f"📂 Résultats sauvegardés dans {CSV_PATH}")

