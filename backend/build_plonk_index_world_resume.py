import os, sys, csv, math, json, glob
import numpy as np
from tqdm import tqdm

# accès au projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plonk_model import run_plonk_api
from sklearn.neighbors import KDTree
import joblib

GT_FILE     = "data/mapillary_world/ground_truth_world_matched.csv"
IMAGES_DIR  = "data/mapillary_world/thumbs_clean"
OUT_DIR     = "data/mapillary_world"
OUT_INDEX   = os.path.join(OUT_DIR, "plonk_world_index.joblib")
SHARD_DIR   = os.path.join(OUT_DIR, "features_plonk")  # dossiers de reprise
SHARD_SIZE  = 200

os.makedirs(SHARD_DIR, exist_ok=True)

# charge GT
with open(GT_FILE) as f:
    gt_rows = list(csv.DictReader(f))
gt = {r["filename"]: (float(r["lat"]), float(r["lon"])) for r in gt_rows}

# liste à encoder
todo = [fn for fn in gt.keys() if os.path.exists(os.path.join(IMAGES_DIR, fn))]
todo.sort()
print(f"➡️ Images à encoder : {len(todo)}")

# trouver les shards déjà présents
existing = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
done_indices = set()
for path in existing:
    meta_path = path.replace(".npz", ".json")
    if not os.path.exists(meta_path): continue
    with open(meta_path) as mf:
        meta = json.load(mf)
    done_indices.update(meta.get("indices", []))
print(f"➡️ Déjà encodées (via shards) : {len(done_indices)}")

# encode manquantes, shard par shard
vectors_all, latlons_all, names_all = [], [], []
current_vecs, current_lats, current_names, current_idx = [], [], [], []
count = 0
for i, fn in enumerate(tqdm(todo, desc="Encodage Plonk (resume)")):
    if i in done_indices:
        continue
    res = run_plonk_api(os.path.join(IMAGES_DIR, fn), return_features=True)
    vec = None if not res else res.get("vector")
    if vec is None:
        continue
    v = np.asarray(vec).reshape(-1)  # 1D
    current_vecs.append(v)
    current_lats.append(gt[fn])
    current_names.append(fn)
    current_idx.append(i)
    count += 1

    if len(current_vecs) >= SHARD_SIZE:
        shard_id = f"{len(existing) + math.ceil(count/SHARD_SIZE):05d}"
        np.savez(os.path.join(SHARD_DIR, f"shard_{shard_id}.npz"),
                 X=np.stack(current_vecs, axis=0),
                 Y=np.asarray(current_lats),
                 N=np.asarray(current_names))
        with open(os.path.join(SHARD_DIR, f"shard_{shard_id}.json"), "w") as mf:
            json.dump({"indices": current_idx}, mf)
        current_vecs, current_lats, current_names, current_idx = [], [], [], []

# flush dernier shard
if current_vecs:
    shard_id = f"{len(existing) + math.ceil(count/SHARD_SIZE):05d}"
    np.savez(os.path.join(SHARD_DIR, f"shard_{shard_id}.npz"),
             X=np.stack(current_vecs, axis=0),
             Y=np.asarray(current_lats),
             N=np.asarray(current_names))
    with open(os.path.join(SHARD_DIR, f"shard_{shard_id}.json"), "w") as mf:
        json.dump({"indices": current_idx}, mf)

# agrégation de TOUS les shards (anciens + nouveaux)
all_npz = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
assert all_npz, "❌ Aucun shard trouvé."
X_list, Y_list, N_list = [], [], []
for p in all_npz:
    z = np.load(p, allow_pickle=True)
    X_list.append(z["X"])
    Y_list.append(z["Y"])
    N_list.append(z["N"])

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)
N = np.concatenate(N_list, axis=0)
print(f"✅ Agrégation : X={X.shape}, Y={Y.shape}, N={N.shape}")

# build KDTree
tree = KDTree(X)
payload = {"kdtree": tree, "latlons": Y, "filenames": N.tolist()}
joblib.dump(payload, OUT_INDEX)
print(f"✅ Index Plonk Monde sauvegardé dans {OUT_INDEX}")

