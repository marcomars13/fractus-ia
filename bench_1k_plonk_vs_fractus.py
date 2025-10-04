#!/usr/bin/env python3
import argparse
import csv
import json
import random
import requests
from pathlib import Path
from tqdm import tqdm
from haversine import haversine

# --- API endpoints ---
PLONK_URL = "http://127.0.0.1:8000/infer/plonk"
FRAC_URL = "http://127.0.0.1:8000/infer/plonk_fractus"


def load_gt(csv_file):
    """Charge le ground truth sous forme {filename: (lat, lon)}"""
    gt = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ✅ Fix : CSV contient "id", pas "filename"
            fname = row["id"] + ".jpg"
            gt[fname] = (float(row["lat"]), float(row["lon"]))
    return gt


def predict_api(img_path, url):
    """Appelle l’API backend"""
    with open(img_path, "rb") as f:
        files = {"file": f}
        try:
            r = requests.post(url, files=files, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data[0]["lat"], data[0]["lon"]
        except Exception as e:
            print(f"❌ Erreur API {url} sur {img_path}: {e}")
            return None


def benchmark(img_dir, gt, n=100, use_api=False):
    """Bench sur n images"""
    results = []

    # Sélectionne n images présentes à la fois dans gt et dans img_dir
    imgs = [f for f in Path(img_dir).glob("*.jpg") if f.name in gt]
    if n < len(imgs):
        imgs = imgs[:n]

    for img in tqdm(imgs, desc="Benchmarking"):
        gt_lat, gt_lon = gt[img.name]

        if use_api:
            pred_plonk = predict_api(img, PLONK_URL)
            pred_fractus = predict_api(img, FRAC_URL)
        else:
            pred_plonk, pred_fractus = None, None  # placeholders

        row = {
            "filename": img.name,
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "plonk_lat": None,
            "plonk_lon": None,
            "fractus_lat": None,
            "fractus_lon": None,
            "err_plonk_km": None,
            "err_fractus_km": None,
        }

        if pred_plonk:
            row["plonk_lat"], row["plonk_lon"] = pred_plonk
            row["err_plonk_km"] = haversine((gt_lat, gt_lon), pred_plonk)

        if pred_fractus:
            row["fractus_lat"], row["fractus_lon"] = pred_fractus
            row["err_fractus_km"] = haversine((gt_lat, gt_lon), pred_fractus)

        results.append(row)

    return results


def save_csv(rows, out_file):
    """Sauvegarde résultats détaillés"""
    keys = rows[0].keys() if rows else []
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, summary_file, args):
    """Calcule et sauve les métriques globales"""
    errs_plonk = [r["err_plonk_km"] for r in rows if r["err_plonk_km"] is not None]
    errs_fractus = [r["err_fractus_km"] for r in rows if r["err_fractus_km"] is not None]

    summary = {
        "n": len(rows),
        "img_dir": args.img_dir,
        "gt_csv": args.gt_csv,
        "mean_err_plonk_km": sum(errs_plonk) / len(errs_plonk) if errs_plonk else None,
        "mean_err_fractus_km": sum(errs_fractus) / len(errs_fractus) if errs_fractus else None,
        "median_err_plonk_km": sorted(errs_plonk)[len(errs_plonk)//2] if errs_plonk else None,
        "median_err_fractus_km": sorted(errs_fractus)[len(errs_fractus)//2] if errs_fractus else None,
        "use_api": args.use_api,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--use_api", action="store_true", help="Utiliser l’API backend au lieu des mocks")
    args = parser.parse_args()

    gt = load_gt(args.gt_csv)
    rows = benchmark(args.img_dir, gt, n=args.n, use_api=args.use_api)

    if rows:
        save_csv(rows, args.out)
        summary = summarize(rows, args.summary, args)
        print(f"✅ Résultats détaillés : {args.out}")
        print(f"📊 Résumé : {args.summary}")
        print(json.dumps(summary, indent=2))
    else:
        print("❌ Aucun résultat (dataset vide ou erreurs API).")


if __name__ == "__main__":
    main()

