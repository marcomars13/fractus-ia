import csv
import os

IMG_DIR = "test_images"
GT_FILE = "data/ground_truth_subset.csv"
OUT_FILE = "data/ground_truth_test_subset.csv"

# charger les noms d'images présents dans test_images/
img_names = {f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")}

with open(GT_FILE, newline="") as fin, open(OUT_FILE, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    kept = 0
    for row in reader:
        if row["filename"] in img_names:
            writer.writerow(row)
            kept += 1

print(f"✅ {kept} entrées sauvegardées dans {OUT_FILE}")


