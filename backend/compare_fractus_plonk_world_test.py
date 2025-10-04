import os
import sys
import argparse
import joblib
import csv
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree

# ✅ Fonction haversine intégrée directement
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Fix chemin backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.plonk_model import run_plonk_api

def load_fractus_index(path):
    obj = joblib.load(path)
    if isinstance(obj, tuple) and len(obj) == 2:
        features, coords = obj
        index = KDTree(features, metric="euclidean")
        return features, coords, index
    elif isinstance(obj, tuple) and len(obj) == 3:
        return obj
    else:
        raise ValueError("❌ Format index Fractus non reconnu")

def fractus_predict(query_vec, index, coords):
    try:
        dists, idxs = index.query(query_vec.reshape(1, -1), k=1)
        lat, lon = coords[idxs[0][0]]
        return {"lat": float(lat), "lon": float(lon)}
    except Exception as e:
        print(f"❌ Erreur dans fractus_predict: {e}")
        return {"lat": None, "lon": None}

def benchmark(images_dir, gt_file, output_csv, fractus_index=None, max_images=50):
    print(f"🚀 Benchmark (mode TEST) sur {gt_file}")
    print(f"📂 Chargement Fractus index: {fractus_index or gt_file}")
    features, coords, index = load_fractus_index(fractus_index or gt_file)
    print(f"📂 Fractus index chargé : {features.shape}")

    plonk_obj = joblib.load("data/mapillary_world/plonk_world_index.joblib")
    latlons = plonk_obj["latlons"]
    filenames = plonk_obj["filenames"]
    print(f"📂 Plonk index chargé : {len(latlons)} coordonnées")

    img_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    print(f"📂 {len(img_files)} images détectées")
    img_files = img_files[:max_images]

    results = []
    for img_name in tqdm(img_files, desc="Comparaison"):
        gt_lat, gt_lon = None, None
        try:
            if img_name in filenames:
                idx = filenames.index(img_name)
                gt_lat, gt_lon = latlons[idx]
        except Exception:
            pass

        q_idx = np.random.randint(0, features.shape[0])
        query_vec = features[q_idx]

        pred_fractus = fractus_predict(query_vec, index, coords)
        err_f = None
        if pred_fractus["lat"] is not None and gt_lat is not None:
            err_f = haversine(gt_lat, gt_lon, pred_fractus["lat"], pred_fractus["lon"])

        pred_plonk = run_plonk_api(os.path.join(images_dir, img_name))
        err_p = None
        if pred_plonk and pred_plonk.get("lat") and gt_lat is not None:
            err_p = haversine(gt_lat, gt_lon, pred_plonk["lat"], pred_plonk["lon"])

        pred_combo, err_c = {"lat": None, "lon": None}, None
        if pred_fractus["lat"] is not None and pred_plonk and pred_plonk.get("lat") is not None:
            lat_c = (pred_fractus["lat"] + pred_plonk["lat"]) / 2
            lon_c = (pred_fractus["lon"] + pred_plonk["lon"]) / 2
            err_c = haversine(gt_lat, gt_lon, lat_c, lon_c)
            pred_combo = {"lat": lat_c, "lon": lon_c}
        elif pred_fractus["lat"] is not None:
            pred_combo, err_c = pred_fractus, err_f
        elif pred_plonk and pred_plonk.get("lat") is not None:
            pred_combo, err_c = pred_plonk, err_p

        results.append({
            "id": img_name,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "f_lat": pred_fractus["lat"], "f_lon": pred_fractus["lon"], "f_err_km": err_f,
            "p_lat": pred_plonk.get("lat") if pred_plonk else None,
            "p_lon": pred_plonk.get("lon") if pred_plonk else None,
            "p_err_km": err_p,
            "c_lat": pred_combo["lat"], "c_lon": pred_combo["lon"], "c_err_km": err_c
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Résultats sauvegardés dans {output_csv}")

    f_errs = [r["f_err_km"] for r in results if r["f_err_km"] is not None]
    p_errs = [r["p_err_km"] for r in results if r["p_err_km"] is not None]
    c_errs = [r["c_err_km"] for r in results if r["c_err_km"] is not None]
    print(f"📊 f_err_km : moyenne={np.mean(f_errs):.2f} km | médiane={np.median(f_errs):.2f} km")
    print(f"📊 p_err_km : moyenne={np.mean(p_errs):.2f} km | médiane={np.median(p_errs):.2f} km")
    print(f"📊 c_err_km : moyenne={np.mean(c_errs):.2f} km | médiane={np.median(c_errs):.2f} km")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--gt_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--fractus_index", type=str, default=None)
    args = parser.parse_args()
    benchmark(args.images_dir, args.gt_file, args.output_csv, fractus_index=args.fractus_index)

if __name__ == "__main__":
    main()

