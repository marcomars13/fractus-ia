import os
import csv
import math
import statistics

CSV_PATH = "backend/plonk_vs_fractus.csv"
CSV_OUT = "backend/plonk_vs_fractus_analyse.csv"

# ================================
# 📍 Fonction Haversine
# ================================
def haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux coordonnées GPS."""
    R = 6371.0  # Rayon terrestre en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ================================
# 📂 Lecture CSV
# ================================
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"❌ Fichier CSV introuvable : {CSV_PATH}")

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            plonk_lat = float(row["plonk_lat"])
            plonk_lon = float(row["plonk_lon"])
            fractus_lat = float(row["fractus_lat"])
            fractus_lon = float(row["fractus_lon"])
            dist = haversine(plonk_lat, plonk_lon, fractus_lat, fractus_lon)
            row["distance_km"] = dist
            rows.append(row)
        except Exception:
            continue  # ignore lignes invalides

if not rows:
    raise ValueError("❌ Aucun résultat exploitable dans le CSV.")

# ================================
# 📊 Analyse
# ================================
distances = [r["distance_km"] for r in rows]
print("📊 Analyse comparative Plonk vs Fractus")
print(f"Nombre d'images comparées : {len(distances)}")
print(f"Distance moyenne : {statistics.mean(distances):.2f} km")
print(f"Distance min : {min(distances):.2f} km")
print(f"Distance max : {max(distances):.2f} km")

# ================================
# 💾 Export CSV enrichi
# ================================
fieldnames = list(rows[0].keys())
with open(CSV_OUT, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"📂 Résultats détaillés sauvegardés dans {CSV_OUT}")

