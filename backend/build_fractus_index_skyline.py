import os
import csv
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import KDTree

from fractus_core import extract_vector
from skyline import extract_skyline

RESULTS_DIR = Path("results/fractus_index")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(gt_file):
    gt = {}
    with open(gt_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if not fname.endswith(".jpg"):
                fname += ".jpg"
            gt[fname] = (float(row["lat"]), float(row["lon"]))
    return gt


def build_index(images_dir, gt_file, subset, output_file):
    gt = load_ground_truth(gt_file)
    img_files = list(Path(images_dir).glob("*.jpg"))

    vectors, filenames = [], []
    for img in img_files:
        fname = img.name
        if fname not in gt:
            continue

        # Fractus base
        vec_f = extract_vector(str(img))

        # Skyline
        vec_s = extract_skyline(str(img))

        # Fusion des deux
        vec = np.concatenate([vec_f, vec_s])

        vectors.append(vec)
        filenames.append(fname)

    X = np.array(vectors, dtype=np.float32)
    tree = KDTree(X)

    with open(output_file, "wb") as f:
        pickle.dump({"tree": tree, "filenames": filenames, "vectors": X}, f)

    print(f"✅ Index Fractus+Skyline construit: {output_file} ({len(filenames)} images)")
    print(f"   → Dimension vecteurs: {X.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="test_images")
    parser.add_argument("--gt_file", type=str, default="data/ground_truth_subset.csv")
    args = parser.parse_args()

    build_index(args.images_dir, args.gt_file, "train_skyline",
                RESULTS_DIR / "fractus_index_train_skyline.pkl")
    build_index(args.images_dir, args.gt_file, "test_skyline",
                RESULTS_DIR / "fractus_index_test_skyline.pkl")

    print("✅ Index Skyline train/test créés !")

