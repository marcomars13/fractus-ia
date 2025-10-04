import os, argparse, subprocess, sys

# Liste de fragments Kaggle YFCC100M (~300-500 Mo chacun)
FRAGMENTS = [
    "chenzhangdaydayup/yfcc100m-0",
    "chenzhangdaydayup/yfcc100m-1",
    "chenzhangdaydayup/yfcc100m-2",
    "chenzhangdaydayup/yfcc100m-3",
    "chenzhangdaydayup/yfcc100m-4",
    "chenzhangdaydayup/yfcc100m-5a",
    "chenzhangdaydayup/yfcc100m-5b",
    "chenzhangdaydayup/yfcc100m-5c",
    "chenzhangdaydayup/yfcc100m-5d",
    "chenzhangdaydayup/yfcc100m-5e"
]

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
    ap.add_argument("--out", default="data/yfcc", help="Dossier de sortie")
    ap.add_argument("--limit-gb", type=float, default=20.0, help="Taille max à télécharger (Go)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    total_gb = get_size_gb(args.out)
    print(f"📂 Dossier actuel : {total_gb:.2f} Go déjà présents")

    for frag in FRAGMENTS:
        if total_gb >= args.limit_gb:
            print(f"✅ Objectif {args.limit_gb} Go atteint, arrêt.")
            break

        print(f"🚀 Téléchargement fragment : {frag}")
        cmd = [
            "kaggle", "datasets", "download", "-d", frag, "-p", args.out, "--unzip"
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"❌ Erreur téléchargement {frag} :", e)
            continue

        total_gb = get_size_gb(args.out)
        print(f"📦 Taille cumulée : {total_gb:.2f} Go")

    print("🎉 Téléchargement YFCC terminé.")
    print(f"📂 Taille finale : {total_gb:.2f} Go dans {args.out}")

if __name__ == "__main__":
    main()

