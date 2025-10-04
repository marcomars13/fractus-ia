import os
import csv
import joblib
import numpy as np
from tqdm import tqdm

# 🔧 Assure-toi qu'on peut importer plonk_model depuis la racine
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plonk_model import run_plonk_api


# 📂 Chemins d’entrée/sortie
GT_FILE = "data/mapillary_world/ground_truth_world_matched.csv"
IMAGES_DIR = "data/mapillary_world/thumbs_clean"
OUTPUT_FILE = "data/mapillary_world/plonk_world_index.joblib"


def build_plonk_index(gt_file, images_dir, output_file):
    """
    Construit un KDTree Plonk basé sur les embeddings générés par run_plonk_api.
    """
    latlons = []
    vectors = []
    filenames = []

    with open(gt_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"➡️ Génération des embeddings Plonk pour {len(rows)} images…")

    for row in tqdm(rows, desc="Encodage Plonk"):
        fname = os.path.basename(row["filename"])
        img_path = os.path.join(images_dir, fname)

        if not os.path.exists(img_path):
            continue

        try:
            pred = run_plonk_api(img_path, return_features=True)
            if pred is None or pred.get("vector") is None:
                continue

            vectors.append(pred["vector"])
            latlons.append((float(row["lat"]), float(row["lon"])))
            filenames.append(fname)

        except Exception as e:
            print(f"❌ Erreur sur {fname}: {e}")
            continue

    if not vectors:
        raise RuntimeError("❌ Aucun vecteur généré, vérifie run_plonk_api et pipe.py")

    vectors = np.array(vectors)
    latlons = np.array(latlons)

    print(f"✅ Génération terminée : {len(vectors)} vecteurs encodés")

    # KDTree
    from sklearn.neighbors import KDTree
    tree = KDTree(vectors)

    payload = {
        "kdtree": tree,
        "latlons": latlons,
        "filenames": filenames,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(payload, output_file)
    print(f"✅ Index Plonk Monde sauvegardé dans {output_file}")


if __name__ == "__main__":
    build_plonk_index(GT_FILE, IMAGES_DIR, OUTPUT_FILE)

