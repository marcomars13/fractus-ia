import json
import csv
import math
import os
import sys

# 📂 Fichiers d'entrée
RESULTS_FILE = "results_10.json"
GT_FILE = "data/ground_truth.csv"  # adapte si ton ground truth est ailleurs

# Vérif fichiers
if not os.path.exists(RESULTS_FILE):
    print(f"❌ Fichier introuvable: {RESULTS_FILE}")
    sys.exit(1)
if not os.path.exists(GT_FILE):
    print(f"❌ Fichier introuvable: {GT_FILE}")
    sys.exit(1)

# 📌 Fonction distance haversine en km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# 📂 Charger ground truth
ground_truth = {}
with open(GT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row["filename"]] = (
            float(row["lat"]),
            float(row["lon"])
        )

# 📂 Charger résultats
with open(RESULTS_FILE) as f:
    results = json.load(f)

errors_fractus = []
errors_plonk = []

for item in results:
    # 🔑 Enlève l'extension .jpg pour matcher avec le ground truth
    fname = item["filename"].replace(".jpg", "")
    if fname not in ground_truth:
        print(f"⚠️ Pas de ground truth pour {fname}, ignoré.")
        continue

    gt_lat, gt_lon = ground_truth[fname]
    frac_lat, frac_lon = item["fractus"]["lat"], item["fractus"]["lon"]
    plonk_lat, plonk_lon = item["plonk"]["lat"], item["plonk"]["lon"]

    err_frac = haversine(gt_lat, gt_lon, frac_lat, frac_lon)
    err_plonk = haversine(gt_lat, gt_lon, plonk_lat, plonk_lon)

    errors_fractus.append(err_frac)
    errors_plonk.append(err_plonk)


# 📊 Résultats
if errors_fractus and errors_plonk:
    mean_frac = sum(errors_fractus) / len(errors_fractus)
    mean_plonk = sum(errors_plonk) / len(errors_plonk)
    gain = (mean_plonk - mean_frac) / mean_plonk * 100

    print("📊 Benchmark sur", len(errors_fractus), "images")
    print(f"Erreur moyenne Plonk   : {mean_plonk:.2f} km")
    print(f"Erreur moyenne Fractus : {mean_frac:.2f} km")
    print(f"🎯 Gain de Fractus     : {gain:.1f} %")
else:
    print("❌ Pas de données exploitables (vérifie ground_truth.csv)")

