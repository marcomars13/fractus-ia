#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit un KDTree Plonk à partir d'un fichier .hdf5 de features.
"""

import os, sys, argparse, joblib, h5py
import numpy as np
from sklearn.neighbors import KDTree

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="features_plonk_xxx.hdf5")
    p.add_argument("--output", required=True, help="fichier .joblib pour sauvegarder l'index")
    args = p.parse_args()

    print(f"📂 Chargement des features depuis {args.features}")
    with h5py.File(args.features, "r") as f:
        feats = f["features"][:]
        ids = f["ids"][:]

    print(f"✅ Chargé {feats.shape[0]} embeddings de dim {feats.shape[1]}")
    print("⚡ Construction KDTree ...")
    tree = KDTree(feats, leaf_size=40, metric="euclidean")

    out = {
        "kdtree": tree,
        "features": feats,
        "ids": ids
    }
    joblib.dump(out, args.output)
    print(f"✨ KDTree sauvegardé dans {args.output}")

if __name__ == "__main__":
    main()

