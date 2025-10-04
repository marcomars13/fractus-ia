import os, argparse, subprocess, sys

DATASET = "habedi/large-dataset-of-geotagged-images"

def get_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/yfcc_good", help="Dossier de sortie")
    ap.add_argument("--limit-gb", type=float, default=20.0, help="Taille max (Go)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"🚀 Téléchargement dataset YFCC : {DATASET}")
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", args.out, "--unzip"]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("❌ Erreur téléchargement Kaggle :", e)
        sys.exit(1)

    total_gb = get_size_gb(args.out)
    print(f"📦 Taille actuelle : {total_gb:.2f} Go")

    if total_gb > args.limit_gb:
        print(f"✂️ Tronquage à {args.limit_gb} Go")
        keep_bytes = args.limit_gb * (1024**3)
        kept = 0
        for root, _, files in os.walk(args.out):
            for f in sorted(files):
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp)
                if kept + sz > keep_bytes:
                    os.remove(fp)
                else:
                    kept += sz
        print(f"✅ Taille finale ≈ {args.limit_gb} Go")

    print("🎉 Téléchargement YFCC terminé")

if __name__ == "__main__":
    main()

