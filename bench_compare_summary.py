#!/usr/bin/env python3
import argparse
import json
import numpy as np
from haversine import haversine

def safe_coord(row, key_lat, key_lon):
    """Récupère un couple (lat, lon) depuis un dictionnaire JSON."""
    return float(row[key_lat]), float(row[key_lon])

def main():
    parser = argparse.ArgumentParser(description="Résumé d’un benchmark Plonk vs Fractus")
    parser.add_argument("--input", required=True, help="Fichier JSON des résultats détaillés")
    parser.add_argument("--out", required=True, help="Résumé JSON de sortie")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    errors_plonk, errors_fractus = [], []

    for row in data:
        # Champs ground truth
        if "gt_lat" in row and "gt_lon" in row:
            gt = safe_coord(row, "gt_lat", "gt_lon")
        elif "lat" in row and "lon" in row:
            gt = safe_coord(row, "lat", "lon")
        else:
            continue  # ligne inutilisable

        # Champs Plonk
        if "pred_lat" in row and "pred_lon" in row:
            pred_plonk = safe_coord(row, "pred_lat", "pred_lon")
            errors_plonk.append(haversine(gt, pred_plonk))

        # Champs Fractus
        if "pred_fractus_lat" in row and "pred_fractus_lon" in row:
            pred_fractus = safe_coord(row, "pred_fractus_lat", "pred_fractus_lon")
            errors_fractus.append(haversine(gt, pred_fractus))

    summary = {
        "n": len(data),
        "n_valid_plonk": len(errors_plonk),
        "n_valid_fractus": len(errors_fractus),
        "mean_err_plonk_km": float(np.mean(errors_plonk)) if errors_plonk else None,
        "median_err_plonk_km": float(np.median(errors_plonk)) if errors_plonk else None,
        "mean_err_fractus_km": float(np.mean(errors_fractus)) if errors_fractus else None,
        "median_err_fractus_km": float(np.median(errors_fractus)) if errors_fractus else None,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("📊 Résumé sauvegardé dans", args.out)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

