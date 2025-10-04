#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_compare_full.py
Compare Plonk, Fractus 1.0, Fractus 8 cœurs et Fractus Full sur un échantillon aléatoire de la base Mapillary.
"""

import math, json, csv, random, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# === PARAMÈTRES ===
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
GROUND_TRUTH_CSV = "data/ground_truth.csv"
OUT_JSON = "results/batch_compare_full.json"
OUT_PNG = "results/batch_compare_full.png"

# === IMPORT DES FONCTIONS ===
try:
    from run_tests import run_plonk, run_plonk_fractus as run_fractus_enriched
    from run_tests_simple import run_plonk_fractus as run_fractus_simple
    from run_tests_full import run_plonk_fractus as run_fractus_full
except ImportError as e:
    print("❌ Impossible d'importer les fonctions nécessaires")
    raise e

# === UTILITAIRES ===
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, newline='', encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fname = row["filename"].strip()
            if not fname.lower().endswith(".jpg"):
                fname = f"{fname}.jpg"
            gt[fname] = (float(row["lat"]), float(row["lon"]))
    return gt

# === MAIN ===
def main(limit=500, seed=42):
    random.seed(seed)
    Path("results").mkdir(parents=True, exist_ok=True)
    gt_map = load_ground_truth(GROUND_TRUTH_CSV)

    all_files = list(gt_map.keys())
    if limit > 0:
        files = random.sample(all_files, min(limit, len(all_files)))
    else:
        files = all_files

    results = []
    plonk_wins = simple_wins = enriched_wins = full_wins = 0

    for idx, fname in enumerate(files, 1):
        img_path = Path(IMG_DIR) / fname
        if not img_path.exists():
            continue

        gt_lat, gt_lon = gt_map[fname]

        try:
            p_lat, p_lon = run_plonk(img_path)
            s_lat, s_lon = run_fractus_simple(img_path)
            e_lat, e_lon = run_fractus_enriched(img_path)
            f_lat, f_lon = run_fractus_full(img_path)
        except Exception as e:
            print(f"❌ Erreur sur {fname}: {e}")
            continue

        dist_p = haversine_km(gt_lat, gt_lon, p_lat, p_lon)
        dist_s = haversine_km(gt_lat, gt_lon, s_lat, s_lon)
        dist_e = haversine_km(gt_lat, gt_lon, e_lat, e_lon)
        dist_f = haversine_km(gt_lat, gt_lon, f_lat, f_lon)

        best = min(dist_p, dist_s, dist_e, dist_f)
        if best == dist_p:
            plonk_wins += 1
        elif best == dist_s:
            simple_wins += 1
        elif best == dist_e:
            enriched_wins += 1
        else:
            full_wins += 1

        results.append({
            "filename": fname,
            "plonk_dist_km": dist_p,
            "fractus_simple_dist_km": dist_s,
            "fractus_enriched_dist_km": dist_e,
            "fractus_full_dist_km": dist_f
        })

        if idx % 50 == 0:
            print(f"🔄 Progression: {idx}/{len(files)} images traitées")

    if not results:
        print("❌ Aucun résultat produit. Vérifie ton dataset.")
        return

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plonk_avg = np.mean([r["plonk_dist_km"] for r in results])
    simple_avg = np.mean([r["fractus_simple_dist_km"] for r in results])
    enriched_avg = np.mean([r["fractus_enriched_dist_km"] for r in results])
    full_avg = np.mean([r["fractus_full_dist_km"] for r in results])
    total = len(results)

    print("\n=== RÉSUMÉ GLOBAL (batch) ===")
    print(f"Plonk : {plonk_wins}/{total} victoires | Erreur moyenne = {plonk_avg:.1f} km")
    print(f"Fractus 1.0 : {simple_wins}/{total} victoires | Erreur moyenne = {simple_avg:.1f} km")
    print(f"Fractus 8 cœurs : {enriched_wins}/{total} victoires | Erreur moyenne = {enriched_avg:.1f} km")
    print(f"Fractus Full : {full_wins}/{total} victoires | Erreur moyenne = {full_avg:.1f} km")

    # Histogramme comparatif
    plt.figure(figsize=(10,6))
    plt.hist([r["plonk_dist_km"] for r in results], bins=50, alpha=0.5, label="Plonk", color="blue")
    plt.hist([r["fractus_simple_dist_km"] for r in results], bins=50, alpha=0.5, label="Fractus 1.0", color="orange")
    plt.hist([r["fractus_enriched_dist_km"] for r in results], bins=50, alpha=0.5, label="Fractus 8 cœurs", color="red")
    plt.hist([r["fractus_full_dist_km"] for r in results], bins=50, alpha=0.5, label="Fractus Full", color="green")
    plt.xlabel("Erreur (km)")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des erreurs - Batch test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print("\n✅ Batch terminé")
    print(f"- Rapport JSON: {OUT_JSON}")
    print(f"- Histogramme: {OUT_PNG}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Nombre d'images à tester (0 = tout)")
    args = parser.parse_args()
    main(limit=args.limit)

