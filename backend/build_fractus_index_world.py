#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit un index Fractus (KDTree + coordonnées) pour le dataset monde.
"""

import os, sys, csv, argparse, joblib
import numpy as np
from sklearn.neighbors import KDTree

# Import fractus_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.fractus_core import extract_vector  # ton encodeur Fractus

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Répertoire des images")
    p.add_argument("--gt-file", required=True, help="Ground truth CSV avec filename,lat,lon")
    p.add_argument("--output", required=True, help="Fichier .joblib pour sauvegarder l’index")
    args = p.parse_args()

    # Charger GT
    gt = {}
    with open(args.gt_file, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if not fname.lower().endswith(".jpg"):
                fname = fname + ".jpg"
            gt[fname] = (float(row["lat"]), float(row["lon"]))

    ids, feats, coords = [], [], []

    print(f"📂 Encodage des images avec Fractus depuis {args.images_dir} ...")
    for fname, (lat, lon) in gt.items():
        fpath = os.path.join(args.images_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            vec = extract_vector(fpath)
            ids.append(fname)
            feats.append(vec)
            coords.append([lat, lon])
        except Exception as e:
            print(f"⚠️ Fractus échoué pour {fname}: {e}")

    feats = np.array(feats, dtype="float32")
    coords = np.array(coords, dtype="float32")

    print(f"✅ Encodé {len(ids)} images, dim={feats.shape[1]}")

    print("⚡ Construction KDTree ...")
    tree = KDTree(feats, leaf_size=40, metric="euclidean")

    out = {
        "kdtree": tree,
        "features": feats,
        "ids": ids,
        "coords": coords,
    }
    joblib.dump(out, args.output)
    print(f"✨ Index Fractus sauvegardé dans {args.output}")

if __name__ == "__main__":
    main()

