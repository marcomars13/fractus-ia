#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, json, random, time
from datetime import datetime
from typing import Dict, Tuple, Any, List

import math
import numpy as np

# ========= Réglages =========
PROJECT_DIR   = "/Users/marco/Projets/fractus-ia"
IMAGES_DIR    = "/Users/marco/mapillary_france_adaptive/thumbs"
GT_CSV        = os.path.join(PROJECT_DIR, "data/ground_truth.csv")
PLONK_DIR     = os.path.join(PROJECT_DIR, "models/plonk_yfcc")  # contient config.json + model.safetensors
SAMPLE_SIZE   = 100   # <- change à 10 / 50 / 100 selon besoin
SEED          = 42

# ========= Préparation des imports =========
sys.path.insert(0, PROJECT_DIR)
# plonk_official a été installé en editable chez toi, mais on sécurise:
sys.path.insert(0, os.path.join(PROJECT_DIR, "plonk_official"))

# ========= Imports locaux =========
try:
    from backend.fallback_fractus import fallback_predict as fractus_predict
except Exception as e:
    print(f"❌ Impossible d'importer Fractus fallback: {e}")
    sys.exit(1)

try:
    from plonk.pipe import PlonkPipeline
    PLONK_OK = True
except Exception as e:
    print(f"❌ Impossible d'importer PlonkPipeline: {e}")
    PLONK_OK = False


# ========= Utilitaires =========
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance haversine en km."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi      = math.radians(lat2 - lat1)
    dlambda   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*r*math.asin(math.sqrt(a))

def load_ground_truth(csv_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Charge le ground truth. Le CSV attendu:
    filename,lat,lon
    1516881052808566,47.63,7.56
    NB: filename peut être sans extension; on normalise sur le 'stem' (sans .jpg)
    """
    gt = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row["filename"]).strip()
            stem = os.path.splitext(name)[0]
            try:
                lat = float(row["lat"]); lon = float(row["lon"])
                gt[stem] = (lat, lon)
            except Exception:
                continue
    return gt

def list_images_in_gt(images_dir: str, gt: Dict[str, Tuple[float,float]]) -> List[str]:
    paths = []
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        if stem in gt:
            paths.append(os.path.join(images_dir, fn))
    return paths

def safe_plonk_predict(pipe, img_path: str) -> Dict[str, Any]:
    """
    Appelle Plonk officiel. Garde une trace lisible en cas d'échec.
    """
    try:
        out = pipe.predict(img_path)
        # Normalise un minimum la sortie attendue
        lat = float(out.get("lat", 0.0)) if isinstance(out, dict) else None
        lon = float(out.get("lon", 0.0)) if isinstance(out, dict) else None
        if lat is None or lon is None:
            return {"ok": False, "error": "format_sortie_inattendu", "raw": repr(out)}
        return {"ok": True, "lat": lat, "lon": lon, "raw": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ========= Main =========
def main():
    random.seed(SEED)

    # 1) Charge GT
    if not os.path.exists(GT_CSV):
        print(f"❌ Ground truth introuvable: {GT_CSV}")
        sys.exit(1)
    gt = load_ground_truth(GT_CSV)
    if not gt:
        print("❌ Ground truth vide/inexploitable.")
        sys.exit(1)

    # 2) Liste images éligibles (présentes dans le GT)
    if not os.path.isdir(IMAGES_DIR):
        print(f"❌ Dossier images introuvable: {IMAGES_DIR}")
        sys.exit(1)
    candidates = list_images_in_gt(IMAGES_DIR, gt)
    if not candidates:
        print("❌ Aucune image du dossier n'a de ground truth correspondant.")
        sys.exit(1)

    # Échantillon
    if SAMPLE_SIZE > len(candidates):
        print(f"ℹ️ Pas assez d'images ({len(candidates)}). On testera sur tout l'ensemble dispo.")
        eval_paths = candidates
    else:
        eval_paths = random.sample(candidates, SAMPLE_SIZE)

    # 3) Initialise Plonk si dispo
    plonk_pipe = None
    if not PLONK_OK:
        print("⚠️ Plonk indisponible à l'import. On évaluera Fractus seulement.")
    else:
        try:
            plonk_pipe = PlonkPipeline(model_path=PLONK_DIR, device="cpu")
            print("✅ Plonk initialisé.")
        except Exception as e:
            print(f"❌ Echec init Plonk: {e}")
            plonk_pipe = None

    results = []
    errs_plonk = []
    errs_fractus = []

    t0 = time.time()
    for idx, img_path in enumerate(eval_paths, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_lat, gt_lon = gt[stem]
        print(f"➡️ [{idx}/{len(eval_paths)}] {os.path.basename(img_path)}")

        # Fractus
        try:
            fr_out = fractus_predict(img_path)
            fr_lat = float(fr_out["lat"]); fr_lon = float(fr_out["lon"])
            err_f = haversine_km(gt_lat, gt_lon, fr_lat, fr_lon)
        except Exception as e:
            fr_out = {"error": str(e)}
            fr_lat = fr_lon = None
            err_f = None

        # Plonk
        if plonk_pipe is not None:
            pl_out = safe_plonk_predict(plonk_pipe, img_path)
            if pl_out.get("ok"):
                pl_lat = pl_out["lat"]; pl_lon = pl_out["lon"]
                err_p = haversine_km(gt_lat, gt_lon, pl_lat, pl_lon)
            else:
                pl_lat = pl_lon = None
                err_p = None
        else:
            pl_out = {"skipped": True}
            pl_lat = pl_lon = None
            err_p = None

        # Accumule
        if err_f is not None: errs_fractus.append(err_f)
        if err_p is not None: errs_plonk.append(err_p)

        results.append({
            "filename": os.path.basename(img_path),
            "gt": {"lat": gt_lat, "lon": gt_lon},
            "fractus": {"lat": fr_lat, "lon": fr_lon, "error_km": err_f, "raw": fr_out},
            "plonk":   {"lat": pl_lat, "lon": pl_lon, "error_km": err_p, "raw": pl_out},
        })

    # 4) Stats
    def mean(x): return float(np.mean(x)) if x else None

    mean_fractus = mean(errs_fractus)
    mean_plonk   = mean(errs_plonk)

    # Gain (%) de Fractus vs Plonk : positif si Fractus plus précis (erreur plus faible)
    if mean_plonk is not None and mean_fractus is not None and mean_plonk > 0:
        gain_pct = (mean_plonk - mean_fractus) / mean_plonk * 100.0
    else:
        gain_pct = None

    # 5) Affiche
    print("\n📊 Résultats")
    if mean_plonk is not None:
        print(f"   Erreur moyenne Plonk   : {mean_plonk:,.2f} km")
    else:
        print("   Erreur moyenne Plonk   : n/a")

    if mean_fractus is not None:
        print(f"   Erreur moyenne Fractus : {mean_fractus:,.2f} km")
    else:
        print("   Erreur moyenne Fractus : n/a")

    if gain_pct is not None:
        print(f"   🎯 Gain de Fractus     : {gain_pct:.1f} %")
    else:
        print("   🎯 Gain de Fractus     : n/a")

    # 6) Sauvegarde JSON
    out_name = f"bench_plonk_vs_fractus_{SAMPLE_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(PROJECT_DIR, out_name)
    payload = {
        "settings": {
            "images_dir": IMAGES_DIR,
            "ground_truth_csv": GT_CSV,
            "plonk_dir": PLONK_DIR,
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
        },
        "summary": {
            "mean_error_plonk_km": mean_plonk,
            "mean_error_fractus_km": mean_fractus,
            "fractus_gain_percent_vs_plonk": gain_pct,
            "num_evaluated": len(eval_paths),
            "num_plonk_ok": len(errs_plonk),
            "num_fractus_ok": len(errs_fractus),
            "duration_sec": round(time.time() - t0, 2),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Détails sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()

