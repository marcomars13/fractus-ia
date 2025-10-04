import csv, joblib, os

PARTIAL = "/Users/marco/Projets/fractus-ia/data/mapillary_world/plonk_world_index_partial.joblib"
GT_IN   = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_matched.csv"
GT_OUT  = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_for_partial.csv"

if not os.path.exists(PARTIAL):
    raise SystemExit("❌ Index partiel introuvable. Lance make_plonk_partial_index.py d'abord.")

payload = joblib.load(PARTIAL)
names = set(payload["filenames"])

kept = 0
with open(GT_IN) as f, open(GT_OUT, "w", newline="") as g:
    rdr = csv.DictReader(f)
    w = csv.writer(g); w.writerow(["filename","lat","lon"])
    for r in rdr:
        if r["filename"] in names:
            w.writerow([r["filename"], r["lat"], r["lon"]]); kept += 1

print(f"✅ GT partiel écrit → {GT_OUT} (lignes conservées : {kept})")

