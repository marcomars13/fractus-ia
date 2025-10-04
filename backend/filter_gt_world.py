#!/usr/bin/env python3
import os, csv

images_dir = "/Users/marco/Projets/fractus-ia/data/mapillary_world"
gt_file = "/Users/marco/Projets/fractus-ia/data/ground_truth_subset.csv"
out_file = "/Users/marco/Projets/fractus-ia/data/ground_truth_world_filtered.csv"

# Récupérer tous les fichiers jpg récursivement
present = set()
for root, _, files in os.walk(images_dir):
    for f in files:
        if f.lower().endswith(".jpg"):
            present.add(os.path.splitext(f)[0])  # sans extension

print(f"📸 Images détectées : {len(present)} fichiers uniques")

with open(gt_file) as fin, open(out_file, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()

    kept = 0
    for row in reader:
        if row["filename"] in present:
            writer.writerow(row)
            kept += 1

print(f"✨ GT filtré écrit dans {out_file} ({kept} lignes gardées)")

