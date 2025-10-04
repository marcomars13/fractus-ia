import os, glob, joblib
import numpy as np
from sklearn.neighbors import KDTree

# ⚡ Chemin absolu pour être sûr
SHARD_DIR = "/Users/marco/Projets/fractus-ia/data/mapillary_world/features_plonk"
OUT_PARTIAL = "/Users/marco/Projets/fractus-ia/data/mapillary_world/plonk_world_index_partial.joblib"

paths = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
if not paths:
    raise SystemExit(f"❌ Aucun shard trouvé dans {SHARD_DIR}")

X_list, Y_list, N_list = [], [], []
for p in paths:
    z = np.load(p, allow_pickle=True)
    X_list.append(z["X"])  # (n_i, D)
    Y_list.append(z["Y"])  # (n_i, 2)
    N_list.append(z["N"])  # (n_i,)

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)
N = np.concatenate(N_list, axis=0)

print(f"✅ Agrégation shards → X={X.shape}, Y={Y.shape}, N={N.shape}")

tree = KDTree(X)
payload = {"kdtree": tree, "latlons": Y, "filenames": N.tolist()}
os.makedirs(os.path.dirname(OUT_PARTIAL), exist_ok=True)
joblib.dump(payload, OUT_PARTIAL)
print(f"✅ Index partiel écrit → {OUT_PARTIAL}")

