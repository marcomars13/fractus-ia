import os
import sys
import csv
import joblib
import numpy as np
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

# Corrige les imports quand lancé depuis /backend ou depuis la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.fractus_core import fractus_predict
from backend.plonk_model import run_plonk_api
from backend.build_fractus_index import load_index_once


# Haversine distance en km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(np.sqrt(a))


def benchmark(images_dir, gt_file, output_csv):
    print(f"🚀 Benchmark Monde avec Fractus vs Plonk sur {images_dir}")

    # Charger index Fractus
    obj = joblib.load(gt_file)
    if isinstance(obj, tuple) and len(obj) == 2:
        features, coords = obj
        from sklearn.neighbors import KDTree
        index = KDTree(features, metric="euclidean")
        print(f"📂 Reconstruction KDTree depuis tuple (features, coords) {features.shape}")
    else:
        features, coords = None, None
        index = obj

    # Liste images
    all_imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
    print(f"📂 {len(all_imgs)} images détectées")

    results = []
    for i, fname in enumerate(tqdm(all_imgs, desc="Traitement")):
        gt_lat, gt_lon = coords[i]

        # Prédiction Fractus
        pred_fractus, err_f = {"lat": None, "lon": None}, None
        try:
            if features is not None:
                pred_fractus = fractus_predict(features[i], index=index) or {"lat": None, "lon": None}
                if pred_fractus["lat"] is not None:
                    err_f = haversine(gt_lat, gt_lon, pred_fractus["lat"], pred_fractus["lon"])
        except Exception as e:
            print(f"❌ Erreur Fractus sur {fname}: {e}")

        # Prédiction Plonk
        pred_plonk, err_p = {"lat": None, "lon": None}, None
        try:
            img_path = os.path.join(images_dir, fname)
            res = run_plonk_api(img_path)
            pred_plonk = res if isinstance(res, dict) else {"lat": None, "lon": None}
            if pred_plonk["lat"] is not None:
                err_p = haversine(gt_lat, gt_lon, pred_plonk["lat"], pred_plonk["lon"])
        except Exception as e:
            print(f"❌ Erreur Plonk sur {fname}: {e}")

        # Combo = Fractus + Plonk si valide, sinon fallback Fractus
        pred_combo, err_c = {"lat": None, "lon": None}, None
        if pred_fractus["lat"] is not None and pred_plonk["lat"] is not None:
            # pondération simple 50/50
            lat_c = (pred_fractus["lat"] + pred_plonk["lat"]) / 2
            lon_c = (pred_fractus["lon"] + pred_plonk["lon"]) / 2
            err_c = haversine(gt_lat, gt_lon, lat_c, lon_c)
            pred_combo = {"lat": lat_c, "lon": lon_c}
        elif pred_fractus["lat"] is not None:
            pred_combo = pred_fractus
            err_c = err_f

        results.append({
            "id": i,
            "filename": fname,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "fractus_lat": pred_fractus["lat"], "fractus_lon": pred_fractus["lon"], "err_fractus": err_f,
            "plonk_lat": pred_plonk["lat"], "plonk_lon": pred_plonk["lon"], "err_plonk": err_p,
            "combo_lat": pred_combo["lat"], "combo_lon": pred_combo["lon"], "err_combo": err_c,
        })

    # Sauvegarde CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Résumé global
    def safe_stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return float("nan"), float("nan")
        return np.mean(vals), np.median(vals)

    avg_f, med_f = safe_stats([r["err_fractus"] for r in results])
    avg_p, med_p = safe_stats([r["err_plonk"] for r in results])
    avg_c, med_c = safe_stats([r["err_combo"] for r in results])

    print("\n📊 Résumé global :")
    print(f"   • Fractus : Moyenne={avg_f:.2f} km | Médiane={med_f:.2f} km")
    print(f"   • Plonk   : Moyenne={avg_p:.2f} km | Médiane={med_p:.2f} km")
    print(f"   • Combo   : Moyenne={avg_c:.2f} km | Médiane={med_c:.2f} km")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--gt_file", required=True)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()
    benchmark(args.images_dir, args.gt_file, args.output_csv)

