import argparse
import pandas as pd
import os

def analyze_failures(csv_file, output_dir="results/failures", top_n=20):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    # Trie par erreur Fractus décroissante
    worst = df.sort_values("fractus_error_km", ascending=False).head(top_n)

    # Sauvegarde CSV des pires cas
    out_csv = os.path.join(
        output_dir,
        os.path.basename(csv_file).replace(".csv", f"_worst{top_n}.csv")
    )
    worst.to_csv(out_csv, index=False)

    print(f"📂 Pires cas sauvegardés : {out_csv}")
    print("\n=== Top erreurs Fractus ===")
    for _, row in worst.iterrows():
        print(
            f"{row['filename']}: "
            f"Fractus {row['fractus_error_km']:.2f} km "
            f"| Plonk {row['plonk_error_km']:.2f} km "
            f"(GT=({row['gt_lat']:.2f},{row['gt_lon']:.2f}))"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV de résultats du benchmark")
    parser.add_argument("--top_n", type=int, default=20, help="Nombre de pires cas à afficher/sauvegarder")
    args = parser.parse_args()

    analyze_failures(args.csv_file, top_n=args.top_n)

