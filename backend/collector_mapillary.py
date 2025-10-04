"""
collector_mapillary.py — télécharge les images Mapillary + vérité terrain
⚡ Version adaptée : télécharge en 1024px et génère ground_truth.csv
"""

import os
import json
import requests
from pathlib import Path
import csv

# 📂 Dossiers de sortie
BASE_DIR = Path("/Users/marco/Projets/fractus-ia/backend/mapillary_out")
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 📂 Fichier JSONL contenant les métadonnées des images
JSONL_FILE = BASE_DIR / "images.jsonl"
GT_FILE = BASE_DIR / "ground_truth.csv"

# ⚡ Fonction de téléchargement
def download_file(url, dest_path):
    try:
        if dest_path.exists():
            return  # déjà téléchargé
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print(f"✅ {dest_path.name}")
        else:
            print(f"⚠️ Erreur {r.status_code} → {url}")
    except Exception as e:
        print(f"❌ Exception {url} → {e}")

print("🚀 Téléchargement Mapillary en cours...")

with open(JSONL_FILE, "r") as fjson, open(GT_FILE, "w", newline="") as fcsv:
    fieldnames = ["filename", "lat", "lon"]
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    for idx, line in enumerate(fjson, 1):
        try:
            data = json.loads(line)

            image_id = data.get("id")
            url = data.get("thumb_1024_url")
            coords = data.get("computed_geometry", {}).get("coordinates")

            if not image_id or not url or not coords:
                continue

            lon, lat = coords  # ⚠️ Mapillary donne [lon, lat]
            filename = f"{image_id}.jpg"
            dest_path = IMAGES_DIR / filename

            # 📥 Télécharger l’image
            download_file(url, dest_path)

            # ✍️ Sauvegarder vérité terrain
            writer.writerow({
                "filename": filename,
                "lat": lat,
                "lon": lon
            })

            if idx % 100 == 0:
                print(f"📊 Progression : {idx} images traitées")

        except Exception as e:
            print(f"❌ Ligne {idx} invalide → {e}")

print("🏁 Collecte terminée")
print(f"📂 Images téléchargées : {IMAGES_DIR}")
print(f"📂 Vérité terrain : {GT_FILE}")

