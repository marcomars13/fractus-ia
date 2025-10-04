import numpy as np
from pathlib import Path
from fractus_core import build_fractus_index

# 📂 dossier résultats
INDEX_DIR = Path("results/fractus_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ Exemple : vecteurs random pour démo (remplace par ton vrai encodeur !)
n_samples = 100
dim = 512
vectors = [np.random.rand(dim).astype(float) for _ in range(n_samples)]

# Split 80% train / 20% test
split = int(0.8 * n_samples)
train_vectors = vectors[:split]
test_vectors = vectors[split:]

# Sauvegarde
build_fractus_index(train_vectors, subset="train")
build_fractus_index(test_vectors, subset="test")

print("✅ Index Fractus train/test créés !")

