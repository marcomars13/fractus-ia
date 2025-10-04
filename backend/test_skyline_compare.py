import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skyline_extractor import extract_skyline_signature

# 📂 Dossier des images Mapillary
img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

if len(files) < 2:
    raise ValueError("❌ Pas assez d'images pour comparer.")

# Prend 2 images différentes
img1 = os.path.join(img_dir, files[0])
img2 = os.path.join(img_dir, files[1])

# Extrait les skylines
sig1 = extract_skyline_signature(img1).reshape(1, -1)
sig2 = extract_skyline_signature(img2).reshape(1, -1)

# Compare via similarité cosinus
sim = cosine_similarity(sig1, sig2)[0][0]

print("🖼️ Image 1:", img1)
print("🖼️ Image 2:", img2)
print(f"🔎 Similarité skyline : {sim:.3f}")

