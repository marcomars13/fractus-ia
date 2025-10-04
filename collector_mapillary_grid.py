#!/usr/bin/env python3
# collector_mapillary_grid.py
# Collecteur Mapillary Monde (échantillon 10k images + miniatures)

import os, json, requests, time
from pathlib import Path

# 📂 Config
OUT_DIR = Path("data/mapillary_world")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "images.jsonl"
OUT_CSV = OUT_DIR / "images.csv"
THUMBS_DIR = OUT_DIR / "thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)  # ✅ création auto du dossier thumbs

LIMIT = 10000   # 🔥 échantillon max
BBOX = [-180, -85, 180, 85]  # Monde entier
GRID_STEP = 10.0             # Taille de la grille (degrés)
ACCESS_TOKEN = os.environ.get("MAPILLARY_TOKEN")

def fetch_images():
    if not ACCESS_TOKEN:
        raise RuntimeError("❌ MAPILLARY_TOKEN non défini. Fais: export MAPILLARY_TOKEN=ton_token")

    collected = 0
    with open(OUT_JSONL, "w") as jf, open(OUT_CSV, "w") as cf:
        cf.write("id,lat,lon\n")
        lat = BBOX[1]
        while lat < BBOX[3] and collected < LIMIT:
            lon = BBOX[0]
            while lon < BBOX[2] and collected < LIMIT:
                url = (
                    f"https://graph.mapillary.com/images"
                    f"?access_token={ACCESS_TOKEN}"
                    f"&fields=id,geometry,thumb_256_url"
                    f"&limit=50"
                    f"&bbox={lon},{lat},{lon+GRID_STEP},{lat+GRID_STEP}"
                )
                try:
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    data = r.json().get("data", [])
                    for d in data:
                        img_id = d["id"]
                        coords = d["geometry"]["coordinates"]
                        lat_, lon_ = coords[1], coords[0]
                        jf.write(json.dumps(d) + "\n")
                        cf.write(f"{img_id},{lat_},{lon_}\n")

                        # 📥 Téléchargement miniature
                        thumb_url = d.get("thumb_256_url")
                        if thumb_url:
                            img_path = THUMBS_DIR / f"{img_id}.jpg"
                            if not img_path.exists():  # éviter doublons
                                try:
                                    img = requests.get(thumb_url, timeout=20)
                                    with open(img_path, "wb") as f:
                                        f.write(img.content)
                                except Exception as e:
                                    print(f"⚠️ Erreur téléchargement {img_id}: {e}")

                        collected += 1
                        if collected >= LIMIT:
                            break
                except Exception as e:
                    print("⚠️ Erreur API:", e)
                lon += GRID_STEP
            lat += GRID_STEP
            time.sleep(0.2)  # éviter le flood
    print(f"✅ Collecté {collected} images avec miniatures → {OUT_DIR}")

if __name__ == "__main__":
    fetch_images()

