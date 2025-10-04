#!/usr/bin/env python3
import argparse, joblib, pandas as pd, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fractus-index", required=True, help="Fichier .joblib avec KDTree + features + ids + coords")
    p.add_argument("--output", required=True, help="CSV de sortie (filename,lat,lon)")
    p.add_argument("--limit", type=int, default=100, help="Nombre d'images à échantillonner")
    args = p.parse_args()

    print(f"📂 Chargement de l’index Fractus : {args.fractus_index}")
    data = joblib.load(args.fractus_index)

    ids = data["ids"]
    coords = data["coords"]

    if len(ids) != len(coords):
        raise RuntimeError("❌ Incohérence index : ids et coords n’ont pas la même taille")

    df = pd.DataFrame({
        "filename": ids,
        "lat": coords[:,0],
        "lon": coords[:,1],
    })

    # Échantillonner
    n = min(args.limit, len(df))
    df_sub = df.sample(n=n, random_state=42).reset_index(drop=True)

    df_sub.to_csv(args.output, index=False)
    print(f"✨ Sous-ensemble écrit dans {args.output} ({len(df_sub)} lignes gardées)")

if __name__ == "__main__":
    main()

