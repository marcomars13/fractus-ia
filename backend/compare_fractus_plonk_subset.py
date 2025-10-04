import os
import sys
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
import csv
import math

# --- Fonction haversine intégrée ici ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def load_subset_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_fractus_index(path):
    obj = joblib.load(path)
    if isinstance(obj, tuple) and len(obj) == 2:
        features, coords = obj
        index = KDTree(features, metric="euclidean")
        return features, coords, index
    elif isinstance(obj, tuple) and len(obj) == 3:
        return obj  # déjà (features, coords, index)
    else:
        raise ValueError("❌ Format Fractus index non reconnu")

def load_plonk_index(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "latlons" in obj:
        coords = np.array(obj["latlons"])
        filenames = obj["filenames"]
        return None, coords, filenames
    elif isinstance(obj, tuple) and len(obj) == 2:
        features, coords = obj
        return features, coords, None
    else:
        raise ValueError("❌ Format Plonk index non reconnu")

def fractus_predict(query_vec, index, coords):
    dist, ind = index.query(query_vec.reshape(1, -1), k=1)
    j = ind[0][0]
    return {"lat": coords[j, 0], "lon": coords[j, 1]}

def benchmark(fractus_index, plonk_index, subset_ids, images_dir, output_csv):
    fX, fY, fIndex = load_fractus_index(fractus_index)
    print(f"📂 Fractus index chargé: {fX.shape}")

    pX, pY, pNames = load_plonk_index(plonk_index)
    print(f"📂 Plonk index chargé: {pY.shape}")

    results = []
    for i, img_id in enumerate(tqdm(subset_ids, desc="Comparaison")):
        gt_lat, gt_lon = fY[i]

        pred_f = fractus_predict(fX[i], fIndex, fY)
        err_f = haversine(gt_lat, gt_lon, pred_f["lat"], pred_f["lon"])

        pred_p, err_p = {"lat": None, "lon": None}, None
        if pNames and img_id in pNames:
            j = pNames.index(img_id)
            pred_p = {"lat": pY[j, 0], "lon": pY[j, 1]}
            err_p = haversine(gt_lat, gt_lon, pred_p["lat"], pred_p["lon"])

        pred_c, err_c = {"lat": None, "lon": None}, None
        if pred_p["lat"] is not None:
            lat_c = (pred_f["lat"] + pred_p["lat"]) / 2
            lon_c = (pred_f["lon"] + pred_p["lon"]) / 2
            pred_c = {"lat": lat_c, "lon": lon_c}
            err_c = haversine(gt_lat, gt_lon, lat_c, lon_c)

        results.append({
            "id": img_id,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "f_lat": pred_f["lat"], "f_lon": pred_f["lon"], "f_err_km": err_f,
            "p_lat": pred_p["lat"], "p_lon": pred_p["lon"], "p_err_km": err_p,
            "c_lat": pred_c["lat"], "c_lon": pred_c["lon"], "c_err_km": err_c,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    f_errs = [r["f_err_km"] for r in results if r["f_err_km"] is not None]
    p_errs = [r["p_err_km"] for r in results if r["p_err_km"] is not None]
    c_errs = [r["c_err_km"] for r in results if r["c_err_km"] is not None]

    print("📊 Résumé global :")
    if f_errs:
        print(f"   • Fractus : moyenne={np.mean(f_errs):.2f} km | médiane={np.median(f_errs):.2f} km")
    if p_errs:
        print(f"   • Plonk   : moyenne={np.mean(p_errs):.2f} km | médiane={np.median(p_errs):.2f} km")
    if c_errs:
        print(f"   • Combo   : moyenne={np.mean(c_errs):.2f} km | médiane={np.median(c_errs):.2f} km")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--fractus_index", type=str, required=True)
    parser.add_argument("--plonk_index", type=str, required=True)
    parser.add_argument("--subset_ids", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    subset_ids = load_subset_ids(args.subset_ids)
    benchmark(args.fractus_index, args.plonk_index, subset_ids, args.images_dir, args.output_csv)

if __name__ == "__main__":
    main()

