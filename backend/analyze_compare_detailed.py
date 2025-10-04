import csv, math
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

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
    out_csv = Path("backend/plonk_fractus_errors.csv")
    out_png = Path("backend/plonk_vs_fractus.png")

    if not results_file.exists() or not gt_file.exists():
        print("❌ Fichier manquant (résultats ou ground truth)")
        return

    # Charger ground truth
    gt_map = {}
    with open(gt_file, newline="") as fgt:
        reader = csv.DictReader(fgt)
        for row in reader:
            image_id = row.get("filename") or row.get("image")
            if not image_id:
                continue
            image_id = Path(image_id).stem
            lat, lon = float(row["lat"]), float(row["lon"])
            gt_map[image_id] = (lat, lon)

    detailed = []
    with open(results_file, newline="") as fres:
        reader = csv.DictReader(fres)
        for row in reader:
            img_id = Path(row["image"]).stem
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

            detailed.append({
                "image": row["image"],
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "plonk_lat": plonk_lat,
                "plonk_lon": plonk_lon,
                "fractus_lat": fractus_lat,
                "fractus_lon": fractus_lon,
                "plonk_error_km": d_plonk,
                "fractus_error_km": d_fractus
            })

    if not detailed:
        print("⚠️ Pas de correspondances trouvées")
        return

    # Export CSV détaillé
    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=detailed[0].keys())
        writer.writeheader()
        writer.writerows(detailed)

    # Statistiques globales
    plonk_errors = [d["plonk_error_km"] for d in detailed]
    fractus_errors = [d["fractus_error_km"] for d in detailed]

    print("\n📊 Résumé détaillé")
    print(f"Plonk   → moyenne {statistics.mean(plonk_errors):.2f} km | médiane {statistics.median(plonk_errors):.2f} km")
    print(f"Fractus → moyenne {statistics.mean(fractus_errors):.2f} km | médiane {statistics.median(fractus_errors):.2f} km")

    # Graphe comparatif
    plt.figure(figsize=(12,6))
    x = list(range(len(detailed)))
    plt.bar(x, plonk_errors, label="Plonk", alpha=0.6)
    plt.bar(x, fractus_errors, label="Fractus", alpha=0.6)
    plt.xlabel("Images")
    plt.ylabel("Erreur (km)")
    plt.title("Comparaison Plonk vs Fractus (erreurs km)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

    print(f"\n📂 CSV détaillé sauvegardé dans {out_csv}")
    print(f"📂 Graphe comparatif sauvegardé dans {out_png}")

if __name__ == "__main__":
    main()

