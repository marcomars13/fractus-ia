import csv
import argparse
import numpy as np

def load_ground_truth(gt_file):
    """Charge le ground truth dans un dict filename -> (lat, lon)."""
    gt = {}
    with open(gt_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["filename"]] = (float(row["lat"]), float(row["lon"]))
    return gt

def load_results(results_file):
    """Charge les résultats comparés (incluant fractus_lat/lon)."""
    results = []
    with open(results_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # on suppose que compare_plonk_fractus_ultimate a déjà écrit fractus_lat/fractus_lon
            if "fractus_lat" in row and "fractus_lon" in row:
                results.append({
                    "filename": row["filename"],
                    "fractus_lat": float(row["fractus_lat"]),
                    "fractus_lon": float(row["fractus_lon"]),
                })
    return results

def compute_offset(results, gt):
    """Calcule les deltas moyens entre Fractus et GT."""
    deltas_lat, deltas_lon = [], []
    for r in results:
        fname = r["filename"]
        if fname in gt:
            gt_lat, gt_lon = gt[fname]
            deltas_lat.append(r["fractus_lat"] - gt_lat)
            deltas_lon.append(r["fractus_lon"] - gt_lon)

    if not deltas_lat:
        print("⚠️ Aucun matching trouvé entre résultats et ground truth.")
        return

    mean_lat = np.mean(deltas_lat)
    mean_lon = np.mean(deltas_lon)
    std_lat = np.std(deltas_lat)
    std_lon = np.std(deltas_lon)

    print("\n📊 Analyse de l'offset Fractus :")
    print(f"   → Moyenne Δlat : {mean_lat:.4f}° (écart-type {std_lat:.4f})")
    print(f"   → Moyenne Δlon : {mean_lon:.4f}° (écart-type {std_lon:.4f})")
    print("   (Un biais fixe en lat/lon pourrait être corrigé par un simple offset.)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True, help="CSV avec prédictions Plonk/Fractus")
    parser.add_argument("--gt_file", required=True, help="CSV ground truth (filename, lat, lon)")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt_file)
    results = load_results(args.results_csv)
    compute_offset(results, gt)

