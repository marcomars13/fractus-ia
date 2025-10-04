# backend/fix_gt_for_tests.py
import csv, os

IMG_DIR = "test_images"
GT_FILE = "data/ground_truth_subset.csv"
OUT_FILE = "data/ground_truth_test_subset.csv"

# récupérer les noms des images présentes dans test_images
img_names = {f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}

with open(GT_FILE, newline="") as fin, open(OUT_FILE, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["filename", "lat", "lon"])
    writer.writeheader()
    kept = 0
    for row in reader:
        # normaliser le nom → rajouter .jpg si besoin
        fname = row["filename"]
        if not fname.endswith(".jpg"):
            fname = fname + ".jpg"
        if fname in img_names:
            writer.writerow({
                "filename": fname,
                "lat": row["lat"],
                "lon": row["lon"]
            })
            kept += 1

print(f"✅ {kept} entrées alignées sauvegardées dans {OUT_FILE}")

