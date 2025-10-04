import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skyline_extractor import extract_skyline_signature

# 📂 Dossier des images Mapillary
img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

if len(files) < 2:
    raise ValueError("❌ Pas assez d'images pour comparer.")

# 🔄 Nombre de paires aléatoires à tester
N_PAIRS = 50

results = []

for _ in range(N_PAIRS):
    img1, img2 = random.sample(files, 2)
    path1 = os.path.join(img_dir, img1)
    path2 = os.path.join(img_dir, img2)

    sig1 = extract_skyline_signature(path1).reshape(1, -1)
    sig2 = extract_skyline_signature(path2).reshape(1, -1)

    sim = cosine_similarity(sig1, sig2)[0][0]
    results.append((sim, img1, img2))

# Trie par similarité
results.sort(key=lambda x: x[0], reverse=True)

print("\n🔝 Top 5 paires les plus proches (similarité élevée) :")
for sim, i1, i2 in results[:5]:
    print(f"  {i1} vs {i2} → {sim:.3f}")

print("\n🔻 Top 5 paires les plus éloignées (similarité faible) :")
for sim, i1, i2 in results[-5:]:
    print(f"  {i1} vs {i2} → {sim:.3f}")


