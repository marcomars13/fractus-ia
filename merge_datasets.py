#!/usr/bin/env python3
import os
import shutil
import csv
from pathlib import Path

# 📂 Sources connues
SOURCES = {
    "france_adaptive": "/Users/marco/mapillary_france_adaptive/thumbs",
    "world": "/Users/marco/Projets/fractus-ia/data/mapillary_world/thumbs",
    "world_2k": "/Users/marco/Projets/fractus-ia/data/mapillary_world_2k/thumbs",
    "backend_out": "/Users/marco/Projets/fractus-ia/backend/mapillary_out/thumbs",
    "backend_images": "/Users/marco/Projets/fractus-ia/backend/mapillary_out/images",
}

# 📂 Destination unifiée
DEST = Path("data/mapillary_all/thumbs_clean")
INDEX = DEST.parent / "index_all.csv"

def main():
    DEST.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []

    for label, src_dir in SOURCES.items():
        src = Path(src_dir)
        if not src.exists():
            print(f"⚠️  Source absente : {src}")
            continue

        for img in src.glob("*.jpg"):
            fname = img.name
            if fname in seen:
                # déjà copié
                rows.append([fname, label, "DUPLICATE"])
                continue

            dest_file = DEST / fname
            try:
                shutil.copy2(img, dest_file)
                seen.add(fname)
                rows.append([fname, label, "OK"])
            except Exception as e:
                rows.append([fname, label, f"ERROR {e}"])

    # ✍️ Sauvegarde de l’index
    with open(INDEX, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "source", "status"])
        writer.writerows(rows)

    print(f"✅ Fusion terminée")
    print(f"   → Images copiées : {len(seen)}")
    print(f"   → Index généré : {INDEX}")

if __name__ == "__main__":
    main()

