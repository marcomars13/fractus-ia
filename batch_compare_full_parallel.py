#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# 🔧 Fix chemin pour accéder à backend/
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ✅ Import garantis
from plonk_model import run_plonk_api
from fractus_core import run_fractus_full_api

# ✅ Mock skyline_enhancer si non dispo
try:
    from skyline_enhancer import enhance_skyline
except ImportError:
    def enhance_skyline(features):
        return features  # mock neutre


def process_image(img_path, profile="default"):
    try:
        plonk_pred = run_plonk_api(img_path)

        features = {"plonk_pred": plonk_pred}
        enhanced = enhance_skyline(features)

        fractus_pred = run_fractus_full_api(img_path, profile=profile)

        return {
            "image": os.path.basename(img_path),
            "plonk": plonk_pred,
            "fractus": fractus_pred,
        }
    except Exception as e:
        return {"image": os.path.basename(img_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Plonk vs Fractus")
    parser.add_argument("--img-dir", type=str, required=True, help="Dossier d’images")
    parser.add_argument("--out", type=str, default="results/batch_compare_full_parallel.json")
    parser.add_argument("--profile", type=str, default="default", help="Profil Fractus")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    results = []

    imgs = sorted([p for p in img_dir.glob("*.jpg")])
    print(f"➡️ Dataset utilisé: {img_dir}")
    print(f"➡️ Images détectées: {len(imgs)}")

    for img in tqdm(imgs, desc="Benchmarking"):
        results.append(process_image(img, profile=args.profile))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Résultats sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()

