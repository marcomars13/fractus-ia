import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import KDTree
import joblib

DEF_BASE = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_for_partial_index.joblib"
DEF_OUT  = "/Users/marco/Projets/fractus-ia/results/bench_split8020_fractus.csv"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(np.sqrt(a))

def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_file", default=DEF_BASE)
    p.add_argument("--output_csv", default=DEF_OUT)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    print(f"📦 Chargement: {args.base_file}")
    obj = joblib.load(args.base_file)
    if not (isinstance(obj, tuple) and len(obj) == 2):
        raise TypeError(f"❌ {args.base_file} ne contient pas (features, coords)")
    features, coords = obj
    n, d = features.shape
    print(f"🔢 Dataset: {n} embeddings de dim {d}")

    # Normalisation L2
    features = l2_normalize(features)

    # Split aléatoire 80/20
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = int((1.0 - args.test_ratio) * n)
    train_idx = perm[:split]
    test_idx  = perm[split:]

    X_train, Y_train = features[train_idx], coords[train_idx]
    X_test,  Y_test  = features[test_idx],  coords[test_idx]

    print(f"🧱 Train: {X_train.shape[0]}  |  🧪 Test: {X_test.shape[0]}")
    print("🌲 Construction KDTree (euclidean sur vecteurs normalisés = cosine)…")
    index = KDTree(X_train, metric="euclidean")

    # Query par batch pour aller vite et propre
    k = args.k
    B = 2048
    all_errs = []
    rows = []
    for s in tqdm(range(0, X_test.shape[0], B), desc="🔎 Évaluation"):
        e = min(s + B, X_test.shape[0])
        dist, idx = index.query(X_test[s:e], k=k)  # (bs, k)

        # vote pondéré 1/distance
        invd = 1.0 / np.maximum(dist, 1e-9)        # (bs, k)
        wsum = invd.sum(axis=1, keepdims=True)     # (bs, 1)
        # coords pondérées
        lat_k = Y_train[idx][:,:,0]                # (bs, k)
        lon_k = Y_train[idx][:,:,1]                # (bs, k)
        lat_pred = (lat_k * invd).sum(axis=1) / wsum.ravel()
        lon_pred = (lon_k * invd).sum(axis=1) / wsum.ravel()

        gt_lat = Y_test[s:e, 0]
        gt_lon = Y_test[s:e, 1]

        # Haversine vectorisé
        errs = np.array([haversine(gt_lat[i], gt_lon[i], lat_pred[i], lon_pred[i]) for i in range(e - s)])
        all_errs.extend(errs.tolist())

        for i in range(e - s):
            rows.append({
                "gt_lat": float(gt_lat[i]),
                "gt_lon": float(gt_lon[i]),
                "pred_lat": float(lat_pred[i]),
                "pred_lon": float(lon_pred[i]),
                "error_km": float(errs[i]),
            })

    # Sauvegarde
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    avg, med = float(np.mean(all_errs)), float(np.median(all_errs))
    print(f"✅ Résultats → {args.output_csv}")
    print(f"📊 Moyenne: {avg:.2f} km   |   Médiane: {med:.2f} km")

    # Top 5 pires
    order = np.argsort(all_errs)[-5:][::-1]
    print("🔎 Top 5 pires erreurs :")
    for j in order:
        r = rows[j]
        print(f"   GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) → Pred=({r['pred_lat']:.2f},{r['pred_lon']:.2f}) | {r['error_km']:.2f} km")

if __name__ == "__main__":
    main()

