import os
import sys
import csv
import joblib
import argparse
import math
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree

# Corrige les imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.fractus_core import fractus_predict


# ------------------------------
# Fonction haversine en km
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ------------------------------
# Benchmark comparatif
# ------------------------------
def benchmark(fractus_file, plonk_file, images_dir, output_csv, max_test=None):
    print(f"🚀 Benchmark Fractus vs Plonk vs Combo")

    # Charger Fractus (features + coords)
    features, coords = joblib.load(fractus_file)
    index_fractus = KDTree(features, metric="euclidean")
    print(f"📂 Fractus index chargé : {features.shape}")

    # Charger Plonk (dict avec kdtree, latlons, filenames)
    plonk_obj = joblib.load(plonk_file)
    kdtree_plonk = plonk_obj["kdtree"]
    latlons_plonk = plonk_obj["latlons"]
    filenames_plonk = plonk_obj["filenames"]
    print(f"📂 Plonk index chargé : {latlons_plonk.shape}")

    results = []

    # Sous-échantillonnage optionnel
    img_list = filenames_plonk
    if max_test:
        img_list = img_list[:max_test]
        print(f"⚠️ Mode test rapide : {len(img_list)} images")

    for i, img_name in enumerate(tqdm(img_list, desc="Comparaison")):
        gt_lat, gt_lon = latlons_plonk[i]

        # Fractus
        try:
            qvec = features[i % len(features)]
            pred_fractus = fractus_predict(qvec, index=index_fractus, coords=coords)
            err_f = haversine(gt_lat, gt_lon, pred_fractus["lat"], pred_fractus["lon"])
        except Exception as e:
            pred_fractus, err_f = {"lat": None, "lon": None}, None
            print(f"❌ Fractus error {img_name}: {e}")

        # Plonk
        try:
            dist, ind = kdtree_plonk.query(qvec.reshape(1, -1), k=1)
            nn_idx = ind[0][0]
            pred_plonk = {"lat": float(latlons_plonk[nn_idx][0]), "lon": float(latlons_plonk[nn_idx][1])}
            err_p = haversine(gt_lat, gt_lon, pred_plonk["lat"], pred_plonk["lon"])
        except Exception as e:
            pred_plonk, err_p = {"lat": None, "lon": None}, None
            print(f"❌ Plonk error {img_name}: {e}")

        # Combo (moyenne des deux)
        pred_combo, err_c = {"lat": None, "lon": None}, None
        if pred_fractus.get("lat") is not None and pred_plonk.get("lat") is not None:
            lat_c = (pred_fractus["lat"] + pred_plonk["lat"]) / 2
            lon_c = (pred_fractus["lon"] + pred_plonk["lon"]) / 2
            pred_combo = {"lat": lat_c, "lon": lon_c}
            err_c = haversine(gt_lat, gt_lon, lat_c, lon_c)
        elif pred_fractus.get("lat") is not None:
            pred_combo, err_c = pred_fractus, err_f
        elif pred_plonk.get("lat") is not None:
            pred_combo, err_c = pred_plonk, err_p

        results.append({
            "id": img_name,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "f_lat": pred_fractus["lat"], "f_lon": pred_fractus["lon"], "f_err_km": err_f,
            "p_lat": pred_plonk["lat"], "p_lon": pred_plonk["lon"], "p_err_km": err_p,
            "c_lat": pred_combo["lat"], "c_lon": pred_combo["lon"], "c_err_km": err_c,
        })

    # Sauvegarde CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Résultats sauvegardés dans {output_csv}")

    # Analyse simple
    for label in ["f_err_km", "p_err_km", "c_err_km"]:
        vals = [r[label] for r in results if r[label] is not None]
        if vals:
            print(f"📊 {label} : moyenne={np.mean(vals):.2f} km | médiane={np.median(vals):.2f} km")

    return results


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractus_file", required=True)
    parser.add_argument("--plonk_file", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_test", type=int, default=None)
    args = parser.parse_args()

    benchmark(args.fractus_file, args.plonk_file, args.images_dir, args.output_csv, max_test=args.max_test)


if __name__ == "__main__":
    main()

