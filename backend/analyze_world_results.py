import csv
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """
    Distance haversine en km entre deux points géo.
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

csv_file = "/Users/marco/Projets/fractus-ia/results/bench_compare_world.csv"

plonk_errors, fractus_errors = [], []

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lat_gt = float(row["gt_lat"])
        lon_gt = float(row["gt_lon"])

        # Plonk
        if row["plonk"] and row["plonk"] != "None":
            try:
                # plonk est stocké comme dict → on nettoie
                plonk_str = row["plonk"].replace("'", "\"")
                import json
                plonk = json.loads(plonk_str)
                if "lat" in plonk and "lon" in plonk:
                    e = haversine(lat_gt, lon_gt, plonk["lat"], plonk["lon"])
                    plonk_errors.append(e)
            except Exception as e:
                continue

        # Fractus
        if row["fractus"] and row["fractus"] != "None":
            try:
                fractus_str = row["fractus"].replace("'", "\"")
                import json
                fractus = json.loads(fractus_str)
                if "mean_score" in fractus:
                    fractus_errors.append(float(fractus["mean_score"]))
            except Exception as e:
                continue

avg_plonk = sum(plonk_errors)/len(plonk_errors) if plonk_errors else None
avg_fractus = sum(fractus_errors)/len(fractus_errors) if fractus_errors else None

print("\n📊 Résumé erreurs moyennes (Monde) :")
if avg_plonk is not None:
    print(f"   • Plonk   : {avg_plonk:.2f} km (sur {len(plonk_errors)} images)")
if avg_fractus is not None:
    print(f"   • Fractus : {avg_fractus:.2f} km (sur {len(fractus_errors)} images)")

