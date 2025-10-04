"""
benchmark_gt.py — compare Plonk vs Plonk+Fractus avec vérité terrain (Mapillary)
"""

import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import plonk_infer
from run_tests_full import run_fractus_full_api

# 📂 Dossiers reliés au collector
BASE_DIR = Path("/Users/marco/Projets/fractus-ia/backend/mapillary_out")
IMG_DIR = BASE_DIR / "images"
GT_FILE = BASE_DIR / "ground_truth.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"

RESULTS_DIR.mkdir(exist_ok=True)

# 📌 Charger la vérité terrain
ground_truth = {}
with open(GT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row["filename"]] = (float(row["lat"]), float(row["lon"]))

# 📊 Stats
total = 0
plonk_wins = 0
fractus_wins = 0

def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre 2 points GPS"""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

print(f"📊 Benchmark vérité terrain sur {len(ground_truth)} images")
print("=" * 60)

# ✍️ Ouvrir le CSV des résultats
with open(RESULTS_FILE, "w", newline="") as csvfile:
    fieldnames = [
        "filename",
        "true_lat", "true_lon",
        "plonk_lat", "plonk_lon",
        "fractus_lat", "fractus_lon",
        "dist_plonk", "dist_fractus",
        "winner"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, fname in enumerate(os.listdir(IMG_DIR), 1):
        if fname not in ground_truth:
            continue

        true_lat, true_lon = ground_truth[fname]
        path = IMG_DIR / fname

        try:
            img = Image.open(path).convert("RGB")
            img_arr = np.array(img)

            # 🔹 Prédictions
            plonk_pred = plonk_infer.plonk_predict(img)[0]
            fractus_pred = run_fractus_full_api(img_arr)

            # 🔍 Distances
            dist_plonk = haversine(true_lat, true_lon, plonk_pred["lat"], plonk_pred["lon"])
            dist_fractus = haversine(true_lat, true_lon, fractus_pred["lat"], fractus_pred["lon"])

            total += 1
            if dist_fractus < dist_plonk:
                fractus_wins += 1
                winner = "Fractus"
            else:
                plonk_wins += 1
                winner = "Plonk"

            # 🖨️ Affichage console
            print(f"[{idx}] {fname}")
            print(f"   Vérité terrain → lat={true_lat}, lon={true_lon}")
            print(f"   Plonk          → {plonk_pred}, dist={dist_plonk:.2f} km")
            print(f"   Fractus Full   → {fractus_pred}, dist={dist_fractus:.2f} km")
            print(f"   ✅ Gagnant : {winner}")
            print("-" * 40)

            # ✍️ Écriture CSV
            writer.writerow({
                "filename": fname,
                "true_lat": true_lat, "true_lon": true_lon,
                "plonk_lat": plonk_pred["lat"], "plonk_lon": plonk_pred["lon"],
                "fractus_lat": fractus_pred["lat"], "fractus_lon": fractus_pred["lon"],
                "dist_plonk": f"{dist_plonk:.4f}",
                "dist_fractus": f"{dist_fractus:.4f}",
                "winner": winner
            })

        except Exception as e:
            print(f"[{idx}] ⚠️ Erreur {fname}: {e}")

print("\n📊 Résumé final")
print("=" * 60)
print(f"Plonk gagnant   : {plonk_wins}/{total}")
print(f"Fractus gagnant : {fractus_wins}/{total}")
print(f"📂 Résultats détaillés : {RESULTS_FILE}")

