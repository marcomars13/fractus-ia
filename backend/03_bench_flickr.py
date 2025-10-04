import os, sys, csv, argparse, json
import numpy as np
import pandas as pd
from time import time

# Charger ton Plonk officiel
from plonk_model import run_plonk_api
# Charger ton Fractus
from fractus_core import extract_vector

import pickle
from sklearn.neighbors import KDTree

def load_fractus_index(index_dir):
    vecs = np.load(os.path.join(index_dir, "vectors.npy"))
    coords = np.load(os.path.join(index_dir, "coords.npy"))
    ids = np.load(os.path.join(index_dir, "ids.npy"), allow_pickle=True)
    with open(os.path.join(index_dir, "kdtree.pkl"), "rb") as f:
        tree = pickle.load(f)["tree"]
    return vecs, coords, ids, tree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/flickr/images")
    ap.add_argument("--gt-file", default="data/flickr/index/flickr_test.csv")
    ap.add_argument("--index-dir", default="data/flickr/index")
    ap.add_argument("--output-csv", default="results/bench_compare_flickr.csv")
    ap.add_argument("--limit", type=int, default=1000, help="limiter nombre d’images testées")
    ap.add_argument("--k", type=int, default=5, help="k voisins pour KDTree")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Load GT
    gt = pd.read_csv(args.gt_file)
    if args.limit > 0 and len(gt) > args.limit:
        gt = gt.sample(args.limit, random_state=42).reset_index(drop=True)

    print(f"📂 Test set : {len(gt)} images")

    # Load Fractus index
    vecs, coords, ids, tree = load_fractus_index(args.index_dir)

    results = []
    t0 = time()
    for i, row in gt.iterrows():
        img_path = os.path.join(args.images_dir, row["filename"])
        lat_gt, lon_gt = float(row["lat"]), float(row["lon"])

        # Prediction Plonk
        try:
            plonk_pred = run_plonk_api(img_path)
            lat_p, lon_p = plonk_pred["lat"], plonk_pred["lon"]
        except Exception:
            lat_p, lon_p = None, None

        # Prediction Fractus
        try:
            v = extract_vector(img_path).reshape(1, -1)
            dist, idx = tree.query(v, k=args.k)
            idx0 = idx[0][0]
            lat_f, lon_f = coords[idx0]
        except Exception:
            lat_f, lon_f = None, None

        results.append({
            "filename": row["filename"],
            "lat_gt": lat_gt, "lon_gt": lon_gt,
            "lat_plonk": lat_p, "lon_plonk": lon_p,
            "lat_fractus": lat_f, "lon_fractus": lon_f,
        })

        if (i+1) % 50 == 0:
            print(f"   • {i+1}/{len(gt)} done")

    # Convert en DataFrame
    df = pd.DataFrame(results)

    # Calculer erreurs km
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        if None in [lat1, lon1, lat2, lon2]:
            return None
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    df["plonk_error_km"] = df.apply(lambda r: haversine(r.lat_gt,r.lon_gt,r.lat_plonk,r.lon_plonk), axis=1)
    df["fractus_error_km"] = df.apply(lambda r: haversine(r.lat_gt,r.lon_gt,r.lat_fractus,r.lon_fractus), axis=1)

    df.to_csv(args.output_csv, index=False)
    print(f"✅ Résultats sauvegardés dans {args.output_csv}")

    # Résumé
    plonk_mean = df["plonk_error_km"].dropna().mean()
    fractus_mean = df["fractus_error_km"].dropna().mean()
    print(f"📊 Résumé erreurs moyennes :")
    print(f"   • Plonk   : {plonk_mean:.2f} km")
    print(f"   • Fractus : {fractus_mean:.2f} km")

if __name__ == "__main__":
    main()

