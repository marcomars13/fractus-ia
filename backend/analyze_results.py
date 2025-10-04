import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_results(csv_file, output_dir="results/analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # Lecture CSV
    df = pd.read_csv(csv_file)

    # Distances
    plonk = df["plonk_error_km"].values
    fractus = df["fractus_error_km"].values

    # Fonction stats
    def stats(arr):
        return {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "max": np.max(arr),
            "count": len(arr),
            "zeros": np.sum(arr == 0.0)   # compteur de 0 km exacts
        }

    plonk_stats = stats(plonk)
    fractus_stats = stats(fractus)

    # Résumé texte
    summary_file = os.path.join(
        output_dir,
        os.path.basename(csv_file).replace(".csv", "_analysis.txt")
    )
    with open(summary_file, "w") as f:
        f.write("=== Résumé analyse résultats ===\n\n")
        f.write("Plonk:\n")
        for k, v in plonk_stats.items():
            f.write(f"  {k:>6}: {v:.2f} km\n")
        f.write("\nFractus:\n")
        for k, v in fractus_stats.items():
            f.write(f"  {k:>6}: {v:.2f} km\n")

    print(f"📂 Résumé sauvegardé : {summary_file}")
    print(f"ℹ️ Zéros détectés → Plonk: {plonk_stats['zeros']}, Fractus: {fractus_stats['zeros']}")

    # Histogrammes
    plt.figure(figsize=(10, 6))
    plt.hist(plonk, bins=50, alpha=0.6, label="Plonk")
    plt.hist(fractus, bins=50, alpha=0.6, label="Fractus")
    plt.xlabel("Erreur (km)")
    plt.ylabel("Nombre d'images")
    plt.title("Histogramme des erreurs")
    plt.legend()
    hist_file = os.path.join(
        output_dir,
        os.path.basename(csv_file).replace(".csv", "_hist.png")
    )
    plt.savefig(hist_file)
    plt.close()
    print(f"📊 Histogramme sauvegardé : {hist_file}")

    # CDF (fonction de répartition cumulative)
    plt.figure(figsize=(10, 6))
    for arr, label in [(plonk, "Plonk"), (fractus, "Fractus")]:
        sorted_err = np.sort(arr)
        cdf = np.arange(len(sorted_err)) / len(sorted_err)
        plt.plot(sorted_err, cdf, label=label)
    plt.xlabel("Erreur (km)")
    plt.ylabel("Proportion cumulée")
    plt.title("CDF des erreurs (proportion ≤ seuil)")
    plt.legend()
    cdf_file = os.path.join(
        output_dir,
        os.path.basename(csv_file).replace(".csv", "_cdf.png")
    )
    plt.savefig(cdf_file)
    plt.close()
    print(f"📈 CDF sauvegardée : {cdf_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Fichier CSV de résultats (bench)")
    parser.add_argument("--output_dir", default="results/analysis", help="Dossier de sortie")
    args = parser.parse_args()

    analyze_results(args.csv_file, args.output_dir)

