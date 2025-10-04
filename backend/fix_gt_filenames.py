import csv
import os

input_file = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world.csv"
output_file = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_fixed.csv"

with open(input_file, "r") as fin, open(output_file, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["filename", "lat", "lon"])

    for row in reader:
        fname = row["filename"]
        if not fname.endswith(".jpg"):
            fname = f"{fname}.jpg"
        writer.writerow([fname, row["lat"], row["lon"]])

print(f"✅ Nouveau GT généré : {output_file}")

