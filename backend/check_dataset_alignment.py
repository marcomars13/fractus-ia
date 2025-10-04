import os, csv
import numpy as np

IMG_DIR = "test_images"
GT_FILE = "data/ground_truth_test_subset.csv"
PLONK_INDEX = "plonk_official/models/plonk_index_coords.csv"
FRACTUS_INDEX = "backend/fractus_index_feats.npy"

# 1. Images dispo
img_names = {f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")}
print(f"📂 Images dans {IMG_DIR}: {len(img_names)}")

# 2. GT subset
gt_names = set()
with open(GT_FILE, newline="") as fin:
    reader = csv.DictReader(fin)
    for row in reader:
        fname = row["filename"]
        if not fname.lower().endswith(".jpg"):
            fname = fname + ".jpg"
        gt_names.add(fname)
print(f"📄 Entrées dans GT: {len(gt_names)}")

# 3. Index Plonk
plonk_names = set()
if os.path.exists(PLONK_INDEX):
    with open(PLONK_INDEX, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            plonk_names.add(row["filename"])
print(f"🧭 Index Plonk: {len(plonk_names)}")

# 4. Index Fractus
if os.path.exists(FRACTUS_INDEX):
    feats = np.load(FRACTUS_INDEX)
    print(f"🌀 Index Fractus: {feats.shape[0]} entrées")

# 5. Cross-check
missing_gt = img_names - gt_names
missing_plonk = img_names - plonk_names
print(f"⚠️ Images sans GT: {len(missing_gt)} → {sorted(list(missing_gt))[:5]}")
print(f"⚠️ Images sans PlonkIndex: {len(missing_plonk)} → {sorted(list(missing_plonk))[:5]}")

