"""
batch_run_tests.py — Batch test Plonk vs Plonk+Fractus
"""

import os
import sys
import csv
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import plonk_infer  # Ton wrapper Plonk
from fractus import compute_fractus_scores
from fractus_profile_utils import load_fractus_profile

# ------------------------
# 🔹 Plonk seul
# ------------------------
def run_plonk_api(image_arr):
    return plonk_infer.plonk_predict(image_arr)


# ------------------------
# 🔹 Plonk + Fractus Full (sécurisé)
# ------------------------
def run_fractus_full_api(image_arr, profile):
    try:
        scores = compute_fractus_scores(image_arr, profile["params"])
        if not isinstance(scores, dict):
            scores = {}
    except Exception as e:
        print(f"⚠️ [compute_fractus_scores] Erreur: {e}")
        scores = {}

    res_plonk = run_plonk_api(image_arr)[0]
    lat = res_plonk["lat"] + scores.get("delta_lat", 0.0)
    lon = res_plonk["lon"] + scores.get("delta_lon", 0.0)

    # Clamp dans les bornes valides
    lat = max(min(lat, 90.0), -90.0)
    lon = max(min(lon, 180.0), -180.0)

    return {"lat": lat, "lon": lon, "meta": {"engine": "Plonk+Fractus"}}


# ------------------------
# 🔹 Distance Haversine
# ------------------------
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ------------------------
# 🔹 Batch Runner
# ------------------------
def batch_run(input_dir, output_csv="results/batch_results.csv"):
    profile = load_fractus_profile()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "lat_plonk", "lon_plonk",
                         "lat_fractus", "lon_fractus", "distance_km"])

        images = [os.path.join(input_dir, x) for x in os.listdir(input_dir)
                  if x.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_path in tqdm(images, desc="🖼️ Traitement des images"):
            try:
                image_arr = np.array(Image.open(img_path).convert("RGB"))

                res_plonk = run_plonk_api(image_arr)[0]
                res_fractus = run_fractus_full_api(image_arr, profile)

                dist = haversine(res_plonk["lat"], res_plonk["lon"],
                                 res_fractus["lat"], res_fractus["lon"])

                writer.writerow([
                    os.path.basename(img_path),
                    res_plonk["lat"], res_plonk["lon"],
                    res_fractus["lat"], res_fractus["lon"],
                    round(dist, 2)
                ])
            except Exception as e:
                print(f"⚠️ Erreur sur {img_path}: {e}")

    print(f"✅ Résultats enregistrés dans {output_csv}")


# ------------------------
# 🔹 Main
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_run_tests.py <dossier_images>")
        sys.exit(1)

    batch_run(sys.argv[1])

