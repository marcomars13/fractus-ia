import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import KDTree

BASE_FILE = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_for_partial_index.joblib"
OUTPUT_CSV = "/Users/marco/Projets/fractus-ia/results/bench_loo_fractus.csv"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def main():
    # Charger (features, coords)
    import joblib
    features, coords = joblib.load(BASE_FILE)

    # Normalisation L2
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Construire index
    index = KDTree(features, metric="euclidean")

    errors = []
    results = []

    for i in tqdm(range(features.shape[0]), desc="Leave-One-Out"):
        # Chercher k voisins, enlever self-match
        dist, idx = index.query(features[i].reshape(1, -1), k=5)
        idx_row = idx[0].tolist()
        dist_row = dist[0].tolist()

        # Exclure self
        candidates = [(j, d) for j, d in zip(idx_row, dist_row) if j != i]
        if not candidates:
            continue

        # Meilleur voisin ≠ self
        best_j, best_d = candidates[0]
        pred_lat, pred_lon = coords[best_j]

        gt_lat, gt_lon = coords[i]
        err = haversine(gt_lat, gt_lon, pred_lat, pred_lon)

        errors.append(err)
        results.append({
            "id": i,
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "error_km": err,
        })

    # Sauvegarde CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # Résumé
    avg_error = np.mean(errors)
    med_error = np.median(errors)
    print(f"\n✅ Résultats sauvegardés dans {OUTPUT_CSV}")
    print(f"📊 Moyenne : {avg_error:.2f} km")
    print(f"📊 Médiane : {med_error:.2f} km")

    top_idx = np.argsort(errors)[-5:][::-1]
    print("🔎 Top 5 pires erreurs :")
    for j in top_idx:
        r = results[j]
        print(f"   id={r['id']} | GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) "
              f"→ Pred=({r['pred_lat']:.2f},{r['pred_lon']:.2f}) "
              f"| erreur={r['error_km']:.2f} km")


if __name__ == "__main__":
    main()

