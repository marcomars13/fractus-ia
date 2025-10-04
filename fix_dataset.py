import os
import csv
import shutil
from pathlib import Path

# 📂 Config
CSV_FILE = "data/mapillary_world/images.csv"
THUMBS_DIR = "data/mapillary_world/thumbs"
OUTPUT_CLEAN = "data/mapillary_world/thumbs_clean"
ORPHANS_DIR = "data/mapillary_world/orphans"
MISSING_LIST = "data/mapillary_world/missing.txt"

def main():
    # Charger la liste des fichiers attendus depuis le CSV
    with open(CSV_FILE, newline="") as f:
        reader = csv.reader(f)
        expected = set(row[0] + ".jpg" for row in reader if row)

    # Lister les fichiers réellement présents
    actual = set(os.listdir(THUMBS_DIR))

    # Identifier en trop et manquants
    extra = actual - expected
    missing = expected - actual
    aligned = expected & actual

    print(f"✅ Alignés : {len(aligned)}")
    print(f"❌ En trop : {len(extra)} (déplacés dans {ORPHANS_DIR})")
    print(f"⚠️ Manquants : {len(missing)} (listés dans {MISSING_LIST})")

    # Préparer les dossiers
    os.makedirs(OUTPUT_CLEAN, exist_ok=True)
    os.makedirs(ORPHANS_DIR, exist_ok=True)

    # Déplacer les fichiers en trop
    for fname in extra:
        src = Path(THUMBS_DIR) / fname
        dst = Path(ORPHANS_DIR) / fname
        try:
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"⚠️ Impossible de déplacer {fname}: {e}")

    # Copier les bons fichiers dans thumbs_clean
    for fname in aligned:
        src = Path(THUMBS_DIR) / fname
        dst = Path(OUTPUT_CLEAN) / fname
        if not dst.exists():
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"⚠️ Impossible de copier {fname}: {e}")

    # Sauver la liste des manquants
    with open(MISSING_LIST, "w") as f:
        for fname in sorted(missing):
            f.write(fname + "\n")

    print("✨ Nettoyage terminé.")

if __name__ == "__main__":
    main()

