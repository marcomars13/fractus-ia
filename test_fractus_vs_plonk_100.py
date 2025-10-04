import requests
import random
import os
import json
import sys
import csv

# 📂 Dossier d’images Mapillary
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"

# 🌍 Endpoint API
URL_PLONK = "http://127.0.0.1:8000/infer/plonk"

# 📂 Ground truth (fichier de référence)
GT_FILE = "data/ground_truth.csv"

# Vérif dossier et ground truth
if not os.path.exists(IMG_DIR):
    print(f"❌ Erreur : dossier introuvable {IMG_DIR}")
    sys.exit(1)
if not os.path.exists(GT_FILE):
    print(f"❌ Erreur : ground truth introuvable {GT_FILE}")
    sys.exit(1)

# Charger la liste des images présentes dans ground truth (ajout .jpg)
valid_images = []
with open(GT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["filename"] + ".jpg"
        if os.path.exists(os.path.join(IMG_DIR, fname)):
            valid_images.append(fname)

if len(valid_images) < 100:
    print(f"❌ Seulement {len(valid_images)} images valides trouvées (ground truth + fichier présent)")
    sys.exit(1)

# 🔢 Sélectionne 100 images aléatoires
sample_images = random.sample(valid_images, 100)

results = []

for idx, img in enumerate(sample_images, start=1):
    img_path = os.path.join(IMG_DIR, img)
    print(f"➡️ [{idx}/100] Test {img} ...")

    try:
        with open(img_path, "rb") as f:
            files = {"file": (img, f, "image/jpeg")}
            response = requests.post(URL_PLONK, files=files)

        if response.status_code == 200:
            out = response.json()
            results.append(out)
            print("✅ OK")
        else:
            print("❌ Error", response.status_code, response.text)

    except Exception as e:
        print(f"❌ Exception pour {img}: {e}")

# Sauvegarde résultats
out_file = "results_100.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📊 Résultats enregistrés dans {out_file}")

