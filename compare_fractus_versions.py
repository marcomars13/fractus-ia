#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, json, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# === PARAMÈTRES ===
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
GROUND_TRUTH_CSV = "data/ground_truth.csv"
OUT_JSON = "results/compare_fractus_versions.json"

# Images à tester
IMAGES_TO_TEST = [
    "1516881052808566.jpg",   # Aubagne
    "1982165845268582.jpg",   # Ste-Baume
    "1983632975125869.jpg",   # Cappadoce
    "9914416162010971.jpg",   # Marseille
    "4252088971509081.jpg",   # Aubagne centre
    "2107742459396991.jpg",   # Ste-Baume autre
]

# === IMPORT DES FONCTIONS ===
try:
    from run_tests import run_plonk, run_plonk_fractus as run_fractus_enriched
    from run_tests_simple import run_plonk_fractus as run_fractus_simple
except ImportError as e:
    print("❌ Impossible d'importer les fonctions nécessaires")
    raise e

# === UTILITAIRE ===
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

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
def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    gt_map = load_ground_truth(GROUND_TRUTH_CSV)
    results = []

    # Compteurs pour stats globales
    plonk_wins = simple_wins = enriched_wins = 0

    for fname in IMAGES_TO_TEST:
        if fname not in gt_map:
            print(f"⚠️ Pas de ground truth pour {fname} dans {GROUND_TRUTH_CSV}")
            continue
        img_path = Path(IMG_DIR) / fname
        if not img_path.exists():
            print(f"⚠️ Image non trouvée dans {IMG_DIR}: {fname}")
            continue

        gt_lat, gt_lon = gt_map[fname]

        # Prédictions
        try:
            p_lat, p_lon = run_plonk(img_path)
            s_lat, s_lon = run_fractus_simple(img_path)
            e_lat, e_lon = run_fractus_enriched(img_path)
        except Exception as e:
            print(f"❌ Erreur pendant la prédiction de {fname}: {e}")
            continue

        # Distances
        dist_p = haversine_km(gt_lat, gt_lon, p_lat, p_lon)
        dist_s = haversine_km(gt_lat, gt_lon, s_lat, s_lon)
        dist_e = haversine_km(gt_lat, gt_lon, e_lat, e_lon)

        # Déterminer le gagnant
        best = min(dist_p, dist_s, dist_e)
        if best == dist_p:
            plonk_wins += 1
        elif best == dist_s:
            simple_wins += 1
        else:
            enriched_wins += 1

        results.append({
            "filename": fname,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "plonk_dist_km": dist_p,
            "fractus_simple_dist_km": dist_s,
            "fractus_enriched_dist_km": dist_e
        })

        print(f"{fname}: Plonk {dist_p:.1f} km | Fractus 1.0 {dist_s:.1f} km | Fractus 8 cœurs {dist_e:.1f} km")

    if not results:
        print("\n❌ Aucune image n’a été traitée.")
        print("👉 Vérifie que tes fichiers sont dans BOTH `thumbs/` et `ground_truth.csv`.")
        return

    # Sauvegarde JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Stats globales
    plonk_avg = np.mean([r["plonk_dist_km"] for r in results])
    simple_avg = np.mean([r["fractus_simple_dist_km"] for r in results])
    enriched_avg = np.mean([r["fractus_enriched_dist_km"] for r in results])
    total = len(results)

    print("\n=== RÉSUMÉ GLOBAL ===")
    print(f"Plonk : {plonk_wins}/{total} victoires | Erreur moyenne = {plonk_avg:.1f} km")
    print(f"Fractus 1.0 : {simple_wins}/{total} victoires | Erreur moyenne = {simple_avg:.1f} km")
    print(f"Fractus 8 cœurs : {enriched_wins}/{total} victoires | Erreur moyenne = {enriched_avg:.1f} km")

    # Graphique comparatif
    labels = [r["filename"] for r in results]
    plonk_vals = [r["plonk_dist_km"] for r in results]
    simple_vals = [r["fractus_simple_dist_km"] for r in results]
    enriched_vals = [r["fractus_enriched_dist_km"] for r in results]

    x = range(len(labels))
    plt.figure(figsize=(10,6))
    plt.bar([i-0.25 for i in x], plonk_vals, width=0.25, label="Plonk", color="blue")
    plt.bar(x, simple_vals, width=0.25, label="Fractus 1.0", color="orange")
    plt.bar([i+0.25 for i in x], enriched_vals, width=0.25, label="Fractus 8 cœurs", color="red")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Erreur (km)")
    plt.title("Comparaison Plonk vs Fractus 1.0 vs Fractus 8 cœurs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/compare_fractus_versions.png", dpi=150)
    plt.close()

    print("\n✅ Comparaison terminée")
    print(f"- Rapport JSON: {OUT_JSON}")
    print("- Graphique généré: results/compare_fractus_versions.png")

if __name__ == "__main__":
    main()

