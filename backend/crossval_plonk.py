import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import KDTree

# ⚡ Dossier des features Plonk officiel
BASE_DIR = "/Users/marco/Projets/fractus-ia/data/mapillary_world/features_plonk"
OUTPUT_CSV = "/Users/marco/Projets/fractus-ia/results/bench_crossval_plonk.csv"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def load_shard(path):
    shard = np.load(path)
    X, Y = shard["X"], shard["Y"]  # X=(n,1024), Y=(n,2)
    # ⚡ Normalisation L2 pour stabilité
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, Y


def evaluate_shard(test_X, test_Y, index, coords, k=5):
    """Évalue un shard test contre l’index (leave-shard-out)."""
    errors = []
    results = []

    dist, idx = index.query(test_X, k=k)

    for i in range(test_X.shape[0]):
        # Vote pondéré par 1/distance sur les k voisins
        top = [(coords[j], d) for j, d in zip(idx[i], dist[i])]
        wsum = sum(1.0 / max(d, 1e-9) for _, d in top)
        lat = sum(latlon[0] * (1.0 / max(d, 1e-9)) for latlon, d in top) / wsum
        lon = sum(latlon[1] * (1.0 / max(d, 1e-9)) for latlon, d in top) / wsum

        gt_lat, gt_lon = test_Y[i]
        err = haversine(gt_lat, gt_lon, lat, lon)

        errors.append(err)
        results.append({
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "pred_lat": lat,
            "pred_lon": lon,
            "error_km": err,
        })

    return errors, results


def main():
    shard_files = sorted([os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".npz")])
    if not shard_files:
        raise FileNotFoundError(f"❌ Aucun shard trouvé dans {BASE_DIR}")

    all_results = []
    all_errors = []

    for test_shard in shard_files:
        print(f"🚀 Cross-val (Plonk) : test sur {os.path.basename(test_shard)}")

        # Charger test set
        test_X, test_Y = load_shard(test_shard)

        # Charger train set = tous sauf shard courant
        train_X, train_Y = [], []
        for f in shard_files:
            if f == test_shard:
                continue
            X, Y = load_shard(f)
            train_X.append(X)
            train_Y.append(Y)

        train_X = np.vstack(train_X)
        train_Y = np.vstack(train_Y)

        # Construire index KDTree (euclidean sur vecteurs normalisés = cosine)
        index = KDTree(train_X, metric="euclidean")

        # Évaluer
        errors, results = evaluate_shard(test_X, test_Y, index, train_Y, k=5)
        all_errors.extend(errors)
        all_results.extend(results)

        print(f"   📊 {len(errors)} images, erreur moyenne = {np.mean(errors):.2f} km, médiane = {np.median(errors):.2f} km")

    # Sauvegarde CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Résultats sauvegardés dans {OUTPUT_CSV}")

    # Résumé global
    avg_error = np.mean(all_errors)
    med_error = np.median(all_errors)
    print(f"\n📊 Résumé global Plonk :")
    print(f"   • Moyenne : {avg_error:.2f} km")
    print(f"   • Médiane : {med_error:.2f} km")

    top_idx = np.argsort(all_errors)[-5:][::-1]
    print("🔎 Top 5 pires erreurs :")
    for i in top_idx:
        r = all_results[i]
        print(f"   GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) "
              f"→ Pred=({r['pred_lat']:.2f},{r['pred_lon']:.2f}) "
              f"| erreur={r['error_km']:.2f} km")


if __name__ == "__main__":
    main()

