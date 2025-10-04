import os
import json
import numpy as np
import cv2

from backend.plonk_model import run_plonk_api
from backend.fractus_parallel import fractus_transform_parallel

# Dossier d'images
IMG_DIR = "/Users/marco/Projets/fractus-ia/data/mapillary_world/thumbs"
OUT_FILE = "/Users/marco/Desktop/bench_direct_plonk_fractus_results.json"

def main():
    results = {}
    imgs = sorted(os.listdir(IMG_DIR))[:10]  # 10 premières images
    
    for fname in imgs:
        path = os.path.join(IMG_DIR, fname)
        print(f"➡️ Test {fname}...")

        img = cv2.imread(path)
        if img is None:
            print(f"❌ Impossible de lire {fname}")
            continue

        # --- Plonk ---
        try:
            plonk_res = run_plonk_api(img)
        except Exception as e:
            plonk_res = {"error": str(e)}

        # --- Fractus (multi-résolution parallèle) ---
        try:
            scores = fractus_transform_parallel(img, workers=8, multi=True)
            if scores is not None and len(scores) > 0:
                fractus_res = {
                    "mean_score": float(np.mean(scores)),
                    "len": len(scores)
                }
            else:
                fractus_res = {"error": "No scores"}
        except Exception as e:
            fractus_res = {"error": str(e)}

        # Sauvegarde par image
        results[fname] = {
            "plonk": plonk_res,
            "fractus": fractus_res
        }

    # Export JSON
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Résultats sauvegardés dans {OUT_FILE}")

if __name__ == "__main__":
    main()

