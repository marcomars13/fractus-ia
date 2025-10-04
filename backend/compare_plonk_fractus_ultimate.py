import os
import csv
import argparse
import numpy as np
from plonk_model import run_plonk_api
from fractus_core import run_fractus_api

def load_ground_truth(gt_file):
    gt_map = {}
    with open(gt_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            fname, lat, lon = row[0], float(row[1]), float(row[2])
            gt_map[fname] = (lat, lon)
    return gt_map

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def benchmark(images_dir, gt_file, output_csv, use_skyline=False):
    gt_map = load_ground_truth(gt_file)
    results, plonk_errors, fractus_errors = [], [], []

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    total = len(files)
    print(f"🚀 Lancement benchmark ({total} images)")

    for i, fname in enumerate(files, 1):
        path = os.path.join(images_dir, fname)

        # clé CSV (avec/sans .jpg)
        key = fname
        if key not in gt_map and key.replace(".jpg", "") in gt_map:
            key = key.replace(".jpg", "")
        if key not in gt_map:
            continue

        gt_lat, gt_lon = gt_map[key]
        try:
            plonk_pred = run_plonk_api(path)
            if isinstance(plonk_pred, list) and len(plonk_pred) > 0:
                plonk_pred = plonk_pred[0]

            # passe l'id pour éviter l'auto-match côté Fractus
            query_id = os.path.splitext(fname)[0]
            fractus_pred = run_fractus_api(path, use_skyline=use_skyline, query_id=query_id)
            if isinstance(fractus_pred, list) and len(fractus_pred) > 0:
                fractus_pred = fractus_pred[0]
        except Exception as e:
            print(f"❌ Erreur sur {fname}: {e}")
            continue

        plonk_dist = haversine(gt_lat, gt_lon, plonk_pred["lat"], plonk_pred["lon"])
        fractus_dist = haversine(gt_lat, gt_lon, fractus_pred["lat"], fractus_pred["lon"])

        plonk_errors.append(plonk_dist)
        fractus_errors.append(fractus_dist)

        results.append([fname, gt_lat, gt_lon,
                        plonk_pred["lat"], plonk_pred["lon"], plonk_dist,
                        fractus_pred["lat"], fractus_pred["lon"], fractus_dist])

        print(f"[{i}/{total}] {fname} → Plonk {plonk_dist:.2f} km | Fractus {fractus_dist:.2f} km")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","gt_lat","gt_lon",
                         "plonk_lat","plonk_lon","plonk_error_km",
                         "fractus_lat","fractus_lon","fractus_error_km"])
        writer.writerows(results)

    plonk_avg = np.mean(plonk_errors) if plonk_errors else None
    fractus_avg = np.mean(fractus_errors) if fractus_errors else None

    print("\n📊 Résumé erreurs moyennes :")
    print(f"   • Plonk   : {plonk_avg:.2f} km"   if plonk_avg is not None   else "   • Plonk   : aucune donnée")
    print(f"   • Fractus : {fractus_avg:.2f} km" if fractus_avg is not None else "   • Fractus : aucune donnée")

    summary_file = output_csv.replace(".csv", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Résumé erreurs moyennes\n")
        f.write("=======================\n")
        f.write(f"Plonk   : {plonk_avg:.2f} km\n"   if plonk_avg is not None   else "Plonk   : aucune donnée\n")
        f.write(f"Fractus : {fractus_avg:.2f} km\n" if fractus_avg is not None else "Fractus : aucune donnée\n")

    print(f"\n📂 Résultats sauvegardés dans {output_csv}")
    print(f"📂 Résumé sauvegardé dans {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("gt_file")
    parser.add_argument("output_csv")
    parser.add_argument("--use_skyline", action="store_true")
    args = parser.parse_args()
    benchmark(args.images_dir, args.gt_file, args.output_csv, use_skyline=args.use_skyline)

