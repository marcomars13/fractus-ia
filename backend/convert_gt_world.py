import csv

input_file = "/Users/marco/Projets/fractus-ia/data/mapillary_world/images.csv"
output_file = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world.csv"

with open(input_file, "r") as fin, open(output_file, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["filename", "lat", "lon"])

    for row in reader:
        try:
            filename = row.get("filename") or row.get("image_path") or row.get("id")
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if filename and lat and lon:
                # si c'est un ID brut → ajoute .jpg
                if not filename.endswith(".jpg"):
                    fname = f"{filename}.jpg"
                else:
                    fname = filename.split("/")[-1]
                writer.writerow([fname, lat, lon])
        except Exception:
            continue

print(f"✅ ground_truth_world.csv régénéré → {output_file}")

