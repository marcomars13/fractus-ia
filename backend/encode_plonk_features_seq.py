#!/usr/bin/env python3
import os, sys, glob, h5py, argparse
import numpy as np

# Fix chemin pour trouver backend/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.plonk_model import get_image_embedding

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    images = sorted(glob.glob(os.path.join(args.images_dir, "**", args.pattern), recursive=True))
    images = images[:args.limit]

    # Probe dimension
    feat = get_image_embedding(images[0])
    dim = feat.shape[0]
    f = h5py.File(args.output, "w")
    ds_feat = f.create_dataset("features", (0, dim), maxshape=(None, dim), dtype="float32")
    ds_ids = f.create_dataset("ids", (0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))

    for img in images:
        emb = get_image_embedding(img)
        i = ds_feat.shape[0]
        ds_feat.resize((i+1, dim))
        ds_ids.resize((i+1,))
        ds_feat[i] = emb
        ds_ids[i] = os.path.basename(img)
        print(f"✅ {img} → {emb.shape}")

    f.close()
    print("✨ Fichier écrit :", args.output)

if __name__ == "__main__":
    main()


