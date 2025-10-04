#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des résultats Plonk vs Fractus (monde).
"""

import csv, numpy as np, sys

def main(csv_path):
    plonk_err, fractus_err, rows = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                ep = float(row["err_plonk_km"])
                ef = float(row["err_fractus_km"])
                plonk_err.append(ep)
                fractus_err.append(ef)
                rows.append((row["filename"], ep, ef))
            except:
                pass

    plonk_err, fractus_err = np.array(plonk_err), np.array(fractus_err)

    print("📊 Résumé global :")
    print(f"   • Plonk   : moyenne={np.nanmean(plonk_err):.2f} km | médiane={np.nanmedian(plonk_err):.2f} km")
    print(f"   • Fractus : moyenne={np.nanmean(fractus_err):.2f} km | médiane={np.nanmedian(fractus_err):.2f} km")

    print("\n🔎 Top 5 pires erreurs (Plonk vs Fractus) :")
    worst = sorted(rows, key=lambda r: max(r[1], r[2]), reverse=True)[:5]
    for fn, ep, ef in worst:
        print(f"   {fn} → Plonk {ep:.2f} km | Fractus {ef:.2f} km")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_compare_world.py <compare_world.csv>")
        sys.exit(1)
    main(sys.argv[1])

