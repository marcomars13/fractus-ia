#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, math, json, time, random, glob
from datetime import datetime
from pathlib import Path

# ========== PARAMÈTRES ==========
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
OUT_CSV = "results/batch_plonk_fractus.csv"
SUMMARY_JSON = "results/summary_stats.json"
GROUND_TRUTH_CSV = "data/ground_truth.csv"
MAX_IMAGES = None
RANDOM_SHUFFLE = True
SLEEP_BETWEEN = 0.0
SEUIL_FAILSAFE = 2.0   # facteur 2× pour anti-plantage

# ========== À BRANCHER : tes appels modèles ==========
def run_plonk(image_path):
    # TODO: remplacer par ton vrai appel Plonk
    h = hash(Path(image_path).name) % 10000
    return 40.0 + (h % 1000)/1000.0, 10.0 + ((h//1000) % 1000)/1000.0

def run_plonk_fractus(image_path):
    # TODO: remplacer par ton vrai pipeline Plonk + Fractus
    h = (hash(Path(image_path).name) + 12345) % 10000
    return 40.0 + (h % 1000)/1000.0, 10.0 + ((h//1000) % 1000)/1000.0

# ========== UTILITAIRES ==========
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_ground_truth(path):
    gt = {}
    if not path or not os.path.exists(path):
        return gt
    with open(path, newline='', encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fname = row["filename"].strip()
            if not fname.lower().endswith(".jpg"):
                fname = f"{fname}.jpg"
            lat = float(row["lat"]); lon = float(row["lon"])
            gt[fname] = (lat, lon)
    return gt

def ensure_dirs():
    Path(os.path.dirname(OUT_CSV)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(SUMMARY_JSON)).mkdir(parents=True, exist_ok=True)

def write_csv_header_if_needed():
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                "ts", "filename",
                "plonk_lat","plonk_lon",
                "fractus_lat","fractus_lon",
                "winner",
                "gt_lat","gt_lon",
                "dist_plonk_km","dist_fractus_km"
            ])

def append_result(row):
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(row)

# ========== PIPELINE ==========
def process_image(image_path, gt_map):
    fname = Path(image_path).name
    ts = datetime.utcnow().isoformat()

    # 1) Prédictions
    p_lat, p_lon = run_plonk(image_path)
    f_lat, f_lon = run_plonk_fractus(image_path)

    # 2) Comparaison avec vérité terrain
    gt_lat = gt_lon = None
    dist_p = dist_f = None
    winner = "plonk"

    if fname in gt_map:
        gt_lat, gt_lon = gt_map[fname]
        dist_p = haversine_km(gt_lat, gt_lon, p_lat, p_lon)
        dist_f = haversine_km(gt_lat, gt_lon, f_lat, f_lon)

        # --- Vote hybride avec seuil anti-plantage ---
        if dist_f < dist_p and dist_f <= dist_p * SEUIL_FAILSAFE:
            winner = "fractus"
        elif dist_p <= dist_f:
            winner = "plonk"
        else:
            winner = "tie->plonk"

    # 3) Sauvegarde
    append_result([
        ts, fname,
        f"{p_lat:.6f}", f"{p_lon:.6f}",
        f"{f_lat:.6f}", f"{f_lon:.6f}",
        winner,
        f"{gt_lat:.6f}" if gt_lat else "",
        f"{gt_lon:.6f}" if gt_lon else "",
        f"{dist_p:.3f}" if dist_p else "",
        f"{dist_f:.3f}" if dist_f else "",
    ])

def summarize(csv_path, out_json):
    plonk_better = fractus_better = tie = total = 0
    dists_p, dists_f = [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if not row["dist_plonk_km"] or not row["dist_fractus_km"]:
                continue
            total += 1
            dp = float(row["dist_plonk_km"])
            df = float(row["dist_fractus_km"])
            dists_p.append(dp); dists_f.append(df)

            # Décision avec failsafe
            if df < dp and df <= dp * SEUIL_FAILSAFE:
                fractus_better += 1
            elif dp <= df:
                plonk_better += 1
            else:
                tie += 1

    stats = {
        "total_eval_samples": total,
        "fractus_win_rate": (fractus_better/total) if total else None,
        "plonk_win_rate": (plonk_better/total) if total else None,
        "tie_rate": (tie/total) if total else None,
        "avg_dist_plonk_km": (sum(dists_p)/len(dists_p)) if dists_p else None,
        "avg_dist_fractus_km": (sum(dists_f)/len(dists_f)) if dists_f else None,
        "generated_at": datetime.utcnow().isoformat()
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats

def main():
    ensure_dirs()
    write_csv_header_if_needed()
    gt_map = load_ground_truth(GROUND_TRUTH_CSV)

    images = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        images.extend(glob.glob(os.path.join(IMG_DIR, ext)))

    if RANDOM_SHUFFLE:
        random.shuffle(images)
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]

    print(f"➡️ Tests à lancer: {len(images)} images")
    start = time.time()
    for i, path in enumerate(images, 1):
        try:
            process_image(path, gt_map)
        except Exception as e:
            print(f"[WARN] Échec {Path(path).name}: {e}")
        if SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)
        if i % 50 == 0:
            stats = summarize(OUT_CSV, SUMMARY_JSON)
            print(f"[{i}/{len(images)}] MAJ stats → Fractus win rate: {stats.get('fractus_win_rate')}")

    stats = summarize(OUT_CSV, SUMMARY_JSON)
    dur = time.time() - start
    print("\n=== FIN DE SÉRIE ===")
    print(f"Images traitées: {len(images)}  |  Durée: {dur/60:.1f} min")
    print("Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

