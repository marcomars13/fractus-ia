import os
import csv
import json
import cv2
import numpy as np
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# Import des modules locaux
from plonk.pipe import PlonkPipeline
from backend.fractus import compute_fractus_scores

# --- CONFIG ---
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
GT_FILE = "data/ground_truth.csv"
OUT_FILE = f"bench_plonk_vs_fractus_100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
N_WORKERS = 8

# --- Wrappers pour uniformiser les sorties ---
def wrap_output(raw):
    """Transforme en dict standard {lat, lon, score}."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 2:
        return {"lat": float(raw[0]), "lon": float(raw[1]), "score": 1.0}
    return None

# --- Chargement du ground truth ---
ground_truth = {}
with open(GT_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fid = row["filename"].split(".")[0]
        ground_truth[fid] = (float(row["lat"]), float(row["lon"]))

# --- Initialisation Plonk ---
LOCAL_REPO = "/Users/marco/Projets/fractus-ia/models/plonk_yfcc"
print(f"🚀 Initialisation Plonk depuis {LOCAL_REPO}")
plonk_pipeline = PlonkPipeline(model_path=LOCAL_REPO, device="cpu")
print("✅ Modèle Plonk prêt")

# --- Fonction de traitement ---
def process_image(fid):
    img_path = os.path.join(IMG_DIR, f"{fid}.jpg")
    if not os.path.exists(img_path):
        return None

    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Prédiction Plonk
        plonk_raw = plonk_pipeline.predict(img_path)
        plonk_out = wrap_output(plonk_raw)

        # Prédiction Fractus
        fractus_raw = compute_fractus_scores(img)
        fractus_out = wrap_output(fractus_raw)

        gt = ground_truth.get(fid)
        if not gt:
            return None

        result = {"id": fid, "gt": gt, "plonk": plonk_out, "fractus": fractus_out}
        return result
    except Exception as e:
        print(f"⚠️ Erreur sur {fid}: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    fids = list(ground_truth.keys())[:100]

    results = []
    with mp.Pool(N_WORKERS) as pool:
        for r in tqdm(pool.imap_unordered(process_image, fids), total=len(fids)):
            if r:
                results.append(r)

    # Calcul des erreurs
    def haversine(coord1, coord2):
        import math
        R = 6371
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    plonk_errs, fractus_errs = [], []
    for r in results:
        gt = r["gt"]
        if r["plonk"]:
            plonk_errs.append(haversine(gt, (r["plonk"]["lat"], r["plonk"]["lon"])))
        if r["fractus"]:
            fractus_errs.append(haversine(gt, (r["fractus"]["lat"], r["fractus"]["lon"])))

    mean_plonk = sum(plonk_errs) / len(plonk_errs) if plonk_errs else float("nan")
    mean_fractus = sum(fractus_errs) / len(fractus_errs) if fractus_errs else float("nan")

    gain = (1 - mean_fractus / mean_plonk) * 100 if mean_plonk and mean_plonk > 0 else 0

    print("\n📊 Résultats")
    print(f"   Erreur moyenne Plonk   : {mean_plonk:,.2f} km")
    print(f"   Erreur moyenne Fractus : {mean_fractus:,.2f} km")
    print(f"   🎯 Gain de Fractus     : {gain:.1f} %")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Détails sauvegardés dans {OUT_FILE}")

