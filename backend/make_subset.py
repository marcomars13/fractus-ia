#!/usr/bin/env python3
import os, argparse, pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-file", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()

    # Charger le GT complet
    df = pd.read_csv(args.gt_file)

    # Normaliser en str et .jpg
    df["filename"] = df["filename"].astype(str)
    df["filename"] = df["filename"].apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")

    # Vérifier que les fichiers existent vraiment
    df["exists"] = df["filename"].apply(lambda f: os.path.exists(os.path.join(args.images_dir, f)))
    df_exist = df[df["exists"]].drop(columns=["exists"])

    if df_exist.empty:
        print("❌ Aucun fichier valide trouvé dans GT + dossier images")
        return

    # Limiter
    df_sub = df_exist.sample(n=min(args.limit, len(df_exist)), random_state=42)

    df_sub.to_csv(args.output, index=False)
    print(f"✨ Sous-ensemble écrit dans {args.output} ({len(df_sub)} lignes gardées)")

if __name__ == "__main__":
    main()

