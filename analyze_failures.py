#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import json, os, shutil
from pathlib import Path

CSV_RESULTS = "results/batch_plonk_fractus.csv"
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
OUT_JSON = "results/failure_report.json"
OUT_CSV = "results/failure_cases.csv"
FAIL_IMG_DIR = "results/failsamples"
THRESHOLD_KM = -100.0   # seuil : Fractus pire que Plonk de 100 km ou plus
MAX_EXAMPLES = 50       # nombre d’images copiées pour inspection

def main():
    # Charger le CSV
    df = pd.read_csv(CSV_RESULTS)
    df = df.dropna(subset=["dist_plonk_km","dist_fractus_km"])
    df["dist_plonk_km"] = df["dist_plonk_km"].astype(float)
    df["dist_fractus_km"] = df["dist_fractus_km"].astype(float)
    df["gain_km"] = df["dist_plonk_km"] - df["dist_fractus_km"]

    # Isoler les plantages (gain < seuil négatif)
    failures = df[df["gain_km"] < THRESHOLD_KM].copy()
    failures_sorted = failures.sort_values("gain_km")  # pire d'abord

    # Sauvegarder en CSV
    failures_sorted.to_csv(OUT_CSV, index=False)

    # Rapport résumé
    stats = {
        "total_samples": len(df),
        "total_failures": len(failures_sorted),
        "failure_rate": len(failures_sorted) / len(df) if len(df) else None,
        "worst_gain_km": failures_sorted["gain_km"].min() if not failures_sorted.empty else None,
        "median_gain_failures": failures_sorted["gain_km"].median() if not failures_sorted.empty else None,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Copier quelques images échantillons
    Path(FAIL_IMG_DIR).mkdir(parents=True, exist_ok=True)
    copied = 0
    for fname in failures_sorted["filename"].head(MAX_EXAMPLES):
        src = Path(IMG_DIR) / fname
        if src.exists():
            dst = Path(FAIL_IMG_DIR) / fname
            shutil.copy(src, dst)
            copied += 1

    # Générer histogramme des gains
    plt.figure(figsize=(10,6))
    plt.hist(df["gain_km"], bins=200, color="skyblue", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--", label="égalité")
    plt.axvline(THRESHOLD_KM, color="red", linestyle="--", label=f"seuil {THRESHOLD_KM} km")
    plt.xlabel("Gain Fractus (km) [+ = meilleur que Plonk, - = pire]")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des gains/pertes Fractus vs Plonk")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/histogram_gains.png", dpi=150)
    plt.close()

    print("✅ Analyse terminée")
    print(f"- Total images analysées: {len(df)}")
    print(f"- Plantages détectés: {len(failures_sorted)} (seuil {THRESHOLD_KM} km)")
    print(f"- Rapport: {OUT_JSON}")
    print(f"- CSV détaillé: {OUT_CSV}")
    print(f"- {copied} images copiées dans {FAIL_IMG_DIR}")
    print(f"- Histogramme généré: results/histogram_gains.png")

if __name__ == "__main__":
    main()

