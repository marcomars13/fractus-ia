#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, json, csv
from pathlib import Path
import matplotlib.pyplot as plt

# === PARAMÈTRES ===
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
GROUND_TRUTH_CSV = "data/ground_truth.csv"
OUT_JSON = "results/local_test_report.json"

# Liste des images à tester
IMAGES_TO_TEST = [
    "1516881052808566.jpg",   # Aubagne
    "1982165845268582.jpg",   # Ste-Baume
    "1983632975125869.jpg",   # Cappadoce
]

# === IMPORT DES FONCTIONS VRAIES ===
try:
    from run_tests import run_plonk, run_plonk_fractus
except ImportError as e:
    print("❌ Impossible d'importer run_plonk / run_plonk_fractus depuis run_tests.py")
    raise e

# === UTILITAIRE ===
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, newline='', encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fname = row["filename"].strip()
            if not fname.lower().endswith(".jpg"):
                fname = f"{fname}.jpg"
            gt[fname] = (float(row["lat"]), float(row["lon"]))
    return gt

def plot_prediction(fname, gt_lat, gt_lon, p_lat, p_lon, f_lat, f_lon):
    plt.figure(figsize=(6,6))
    plt.scatter(gt_lon, gt_lat, c="green", marker="*", s=150, label="Vérité terrain")
    plt.scatter(p_lon, p_lat, c="blue", marker="o", s=80, label="Plonk")
    plt.scatter(f_lon, f_lat, c="red", marker="x", s=80, label="Fractus")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Comparaison Plonk vs Fractus\n{fname}")
    plt.legend()
    plt.grid(True)

    out_path = Path("results") / f"map_{Path(fname).stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🗺️ Carte générée: {out_path}")

# === MAIN ===
def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    gt_map = load_ground_truth(GROUND_TRUTH_CSV)
    results = []

    for fname in IMAGES_TO_TEST:
        if fname not in gt_map:
            print(f"⚠️ Pas de ground truth pour {fname}")
            continue
        img_path = Path(IMG_DIR) / fname
        if not img_path.exists():
            print(f"⚠️ Image non trouvée: {img_path}")
            continue

        gt_lat, gt_lon = gt_map[fname]
        p_lat, p_lon = run_plonk(img_path)
        f_lat, f_lon = run_plonk_fractus(img_path)

        dist_p = haversine_km(gt_lat, gt_lon, p_lat, p_lon)
        dist_f = haversine_km(gt_lat, gt_lon, f_lat, f_lon)

        winner = "fractus" if dist_f < dist_p else "plonk" if dist_p < dist_f else "tie"

        results.append({
            "filename": fname,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "plonk_lat": p_lat, "plonk_lon": p_lon, "dist_plonk_km": dist_p,
            "fractus_lat": f_lat, "fractus_lon": f_lon, "dist_fractus_km": dist_f,
            "winner": winner
        })

        # Générer carte
        plot_prediction(fname, gt_lat, gt_lon, p_lat, p_lon, f_lat, f_lon)

    # Sauvegarde JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== LOCAL TEST REPORT ===")
    for r in results:
        print(f"{r['filename']}: Plonk {r['dist_plonk_km']:.1f} km vs Fractus {r['dist_fractus_km']:.1f} km → {r['winner']}")

if __name__ == "__main__":
    main()

