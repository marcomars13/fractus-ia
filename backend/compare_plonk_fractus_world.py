import os
import sys
import csv
import argparse
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# Corrige l'import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.plonk_model import run_plonk_api
from backend.fractus_core import fractus_predict


def benchmark(gt_file, output_csv, plonk_index_path, fractus_index_path, limit=None):
    # --- Charger GT ---
    gt = pd.read_csv(gt_file)
    if limit:
        gt = gt.head(limit)

    # --- Charger index Plonk ---
    print(f"📂 Chargement index Plonk depuis {plonk_index_path}")
    plonk_index = joblib.load(plonk_index_path)

    # --- Charger index Fractus ---
    print(f"📂 Chargement index Fractus depuis {fractus_index_path}")
    fractus_index = joblib.load(fractus_index_path)
    kdtree_f = fractus_index["kdtree"]
    features_f = fractus_index["features"]
    coords_f = fractus_index["coords"]
    ids_f = fractus_index["ids"]

    # Résultats
    results = []

    for _, row in tqdm(gt.iterrows(), total=len(gt), desc="Comparaison"):
        fname = row["filename"]
        lat_gt, lon_gt = row["lat"], row["lon"]

        err_plonk, err_fractus = np.nan, np.nan

        # --- Plonk ---
        try:
            out = run_plonk_api(
                os.path.join(os.path.dirname(gt_file), fname)
            )
            if out and "lat" in out and "lon" in out:
                lat_p, lon_p = out["lat"], out["lon"]
                err_plonk = np.sqrt((lat_p - lat_gt) ** 2 + (lon_p - lon_gt) ** 2) * 111
        except Exception as e:
           

