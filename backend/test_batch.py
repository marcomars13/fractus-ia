"""
test_batch.py — Benchmark auto Plonk vs Plonk+Fractus sur 50 images
"""

import os
import random
import numpy as np
from PIL import Image
import plonk_infer
from run_tests_full import run_fractus_full_api

# 📂 Dossier d'images de test
IMG_DIR = "/Users/marco/Projets/fractus_plonk/plonk_repo/demo/examples"

# Récupère toutes les images du dossier
all_imgs = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

# Tire au hasard 50 (ou moins si dispo)
imgs = random.sample(all_imgs, min(50, len(all_imgs)))

print(f"📊 Test sur {len(imgs)} images...")
print("="*60)

for idx, path in enumerate(imgs, 1):
    try:
        img = Image.open(path).convert("RGB")
        img_arr = np.array(img)

        # 🔹 Plonk seul
        plonk_pred = plonk_infer.plonk_predict(img)

        # 🔹 Plonk + Fractus
        fractus_pred = run_fractus_full_api(img_arr)

        print(f"\n[{idx}] {os.path.basename(path)}")
        print(f"   Plonk      → {plonk_pred}")
        print(f"   FractusFull→ {fractus_pred}")

    except Exception as e:
        print(f"[{idx}] ⚠️ Erreur avec {path} → {e}")


