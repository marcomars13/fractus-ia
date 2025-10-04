import os
import pandas as pd

IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
GT_FILE = "data/ground_truth.csv"
OUT_FILE = "data/ground_truth_subset.csv"

# Charger le gros CSV
gt = pd.read_csv(GT_FILE, dtype={"filename": str})
gt["filename"] = gt["filename"].str.strip().str.replace(".0", "", regex=False)

# Lister images disponibles en local
available = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}

# Garder uniquement celles qui existent
subset = gt[gt["filename"].isin(available)]

print(f"📊 Subset : {len(subset)} lignes gardées sur {len(gt)}")
subset.to_csv(OUT_FILE, index=False)
print(f"📂 CSV sauvegardé dans {OUT_FILE}")

