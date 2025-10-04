import os
import numpy as np
import pandas as pd
import joblib
from build_fractus_index import build_fractus_index

BASE_DIR = "/Users/marco/Projets/fractus-ia/data/mapillary_world"
features_dir = os.path.join(BASE_DIR, "features_plonk")
output_file = os.path.join(BASE_DIR, "ground_truth_world_for_partial_index.joblib")

def main():
    features_list, coords_list = [], []

    feature_files = sorted(
        [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith(".npz")]
    )
    if not feature_files:
        raise FileNotFoundError(f"❌ Aucun .npz trouvé dans {features_dir}")

    for f in feature_files:
        shard = np.load(f)
        X = shard["X"]       # features
        coords_subset = shard["Y"]  # déjà (lat, lon)
        if X.shape[0] != coords_subset.shape[0]:
            raise ValueError(f"❌ Incohérence dans {f} : {X.shape[0]} features vs {coords_subset.shape[0]} coords")

        features_list.append(X)
        coords_list.append(coords_subset)
        print(f"📂 {f} → features {X.shape}, coords {coords_subset.shape}")

    features = np.vstack(features_list)
    coords = np.vstack(coords_list)

    print(f"📊 Features concaténées : {features.shape}")
    print(f"📊 Coords concaténées   : {coords.shape}")

    # Construire et sauvegarder l’index
    tree, coords = build_fractus_index(features, coords, output_file)
    print(f"✅ Nouvel index sauvegardé dans : {output_file}")

    # 🔎 Sanity check : vérifier que KDTree retrouve bien ses propres coordonnées
    dist, idx = tree.query(features[:5], k=1)
    print("🔎 Sanity check KDTree (5 premiers points):")
    for i in range(5):
        lat_pred, lon_pred = coords[idx[i][0]]
        lat_gt, lon_gt = coords[i]
        print(
            f"   id={i} | GT=({lat_gt:.4f},{lon_gt:.4f}) "
            f"→ Pred=({lat_pred:.4f},{lon_pred:.4f}) | dist={dist[i][0]:.6f}"
        )

if __name__ == "__main__":
    main()

