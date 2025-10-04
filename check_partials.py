import os
import json
from pathlib import Path

PARTIAL_DIR = "results/partials"

def check_partials():
    partials = sorted(Path(PARTIAL_DIR).glob("partial_*.json"))
    if not partials:
        print("⚠️ Aucun fichier partial trouvé.")
        return

    total_valid = 0
    total_invalid = 0

    for part in partials:
        with open(part) as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"❌ Impossible de lire {part}: {e}")
                continue

        if not isinstance(data, list):
            print(f"❌ {part} n'est pas une liste JSON")
            continue

        valid = [r for r in data if isinstance(r, dict) and "dists" in r]
        invalid = len(data) - len(valid)

        total_valid += len(valid)
        total_invalid += invalid

        print(f"{part.name}: {len(valid)} valides, {invalid} corrompues")

    print("\n=== RÉSUMÉ GLOBAL ===")
    print(f"✔️ Total valides   : {total_valid}")
    print(f"⚠️ Total corrompus : {total_invalid}")
    print(f"📂 Fichiers scannés: {len(partials)}")

if __name__ == "__main__":
    check_partials()

