# nano /Users/marco/Projets/fractus-ia/backend/train_fractus_subset.py

import os
import argparse
import joblib
import numpy as np
from sklearn.neighbors import KDTree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, required=True,
                        help="Index global Fractus (features, coords)")
    parser.add_argument("--subset_ids", type=str, required=True,
                        help="Fichier texte contenant 1 nom de fichier/image_id par ligne")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Nom du nouvel index spécialisé")
    args = parser.parse_args()

    print(f"📦 Chargement index global: {args.index_file}")
    features, coords = joblib.load(args.index_file)

    print(f"📂 Chargement subset IDs: {args.subset_ids}")
    with open(args.subset_ids, "r") as f:
        wanted = [line.strip() for line in f if line.strip()]

    # supposer que les IDs correspondent aux positions dans l’index (0..N-1)
    # ou bien directement aux filenames si tu avais sauvegardé la liste
    # ici je prends la version "indices entiers"
    wanted_idx = [int(x) for x in wanted]

    sub_features = features[wanted_idx]
    sub_coords = coords[wanted_idx]

    print(f"🔢 Subset: {sub_features.shape[0]} échantillons")

    print("🌲 Construction KDTree subset...")
    tree = KDTree(sub_features, metric="euclidean")

    joblib.dump((sub_features, sub_coords, tree), args.output_file)
    print(f"✅ Mini-index sauvegardé dans: {args.output_file}")

if __name__ == "__main__":
    main()

