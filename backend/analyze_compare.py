import csv, math
from pathlib import Path
import statistics

# 📍 Fonction distance haversine (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def main():
    results_file = Path("backend/plonk_fractus_full_results.csv")
    gt_file = Path("data/ground_truth_subset.csv")

    if not results_file.exists():
        print(f"❌ Fichier résultats manquant: {results_file}")
        return
    if not gt_file.exists():
        print(f"❌ Fichier ground truth manquant: {gt_file}")
        return

    # Charger ground truth dans un dict {image_id: (lat, lon)}
    gt_map = {}
    with open(gt_file, newline="") as fgt:
        reader = csv.DictReader(fgt)
        for row in reader:
            image_id = row.get("filename") or row.get("image")
            if not image_id:
                continue
            image_id = Path(image_id).stem  # 🔑 on enlève extension
            lat, lon = float(row["lat"]), float(row["lon"])
            gt_map[image_id] = (lat, lon)

    # Charger résultats
    plonk_dists, fractus_dists = [], []
    plonk_wins, fractus_wins = 0, 0
    with open(results_file, newline="") as fres:
        reader = csv.DictReader(fres)
        for row in reader:
            img_id = Path(row["image"]).stem  # 🔑 harmonisé avec GT
            if img_id not in gt_map:
                continue
            gt_lat, gt_lon = gt_map[img_id]

            try:
                plonk_lat, plonk_lon = float(row["plonk_lat"]), float(row["plonk_lon"])
                fractus_lat, fractus_lon = float(row["fractus_lat"]), float(row["fractus_lon"])
            except:
                continue

            d_plonk = haversine(gt_lat, gt_lon, plonk_lat, plonk_lon)
            d_fractus = haversine(gt_lat, gt_lon, fractus_lat, fractus_lon)

            plonk_dists.append(d_plonk)
            fractus_dists.append(d_fractus)

            if d_plonk < d_fractus:
                plonk_wins += 1
            elif d_fractus < d_plonk:
                fractus_wins += 1

    # Résumé
    print("\n📊 Résumé comparatif Plonk vs Fractus")
    if plonk_dists and fractus_dists:
        print(f"Plonk   → moyenne {statistics.mean(plonk_dists):.2f} km | médiane {statistics.median(plonk_dists):.2f} km")
        print(f"Fractus → moyenne {statistics.mean(fractus_dists):.2f} km | médiane {statistics.median(fractus_dists):.2f} km")
        print(f"🏆 Victoires: Plonk={plonk_wins} | Fractus={fractus_wins}")
    else:
        print("⚠️ Pas de données comparables trouvées (vérifie ground truth et résultats).")

if __name__ == "__main__":
    main()

