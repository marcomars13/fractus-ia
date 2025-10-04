import os, sys, csv, random, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plonk_model import run_plonk_api
from sklearn.neighbors import KDTree

IMAGES_DIR = "data/mapillary_world/thumbs_clean"
GT_FILE    = "data/mapillary_world/ground_truth_world_matched.csv"
N = 50

# charge GT
with open(GT_FILE) as f:
    gt = {r["filename"]:(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)}

imgs = [fn for fn in os.listdir(IMAGES_DIR) if fn in gt]
random.shuffle(imgs)
subset = imgs[:N]

vecs, latlons = [], []
for fn in subset:
    res = run_plonk_api(os.path.join(IMAGES_DIR, fn), return_features=True)
    if not res or res.get("vector") is None: continue
    v = np.asarray(res["vector"]).reshape(-1)   # 1D
    vecs.append(v)
    latlons.append(gt[fn])

assert len(vecs) >= 10, f"Pas assez de vecteurs ({len(vecs)})"
X = np.stack(vecs, axis=0)  # (N,D)
print("✅ X shape:", X.shape)

tree = KDTree(X)
print("✅ KDTree OK sur sample")


