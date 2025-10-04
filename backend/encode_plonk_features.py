#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode Plonk features — multi-core, HDF5 extensible.
Corrigé pour éviter les soucis de pickle (get_emb défini globalement).
"""

import os, sys, glob, h5py, time, argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List
import numpy as np

# Corrige l'import (racine ou dossier /backend)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import direct du wrapper Plonk
from backend.plonk_model import get_image_embedding

# ---------------- Worker ---------------- #

def _safe_rel_id(root: str, path: str) -> str:
    try:
        rel = os.path.relpath(path, root)
        return rel if not rel.startswith("..") else os.path.basename(path)
    except Exception:
        return os.path.basename(path)

def _process_one(path, root, include_x=False):
    img_id = _safe_rel_id(root, path)
    try:
        f = get_image_embedding(path)
        if f is None:
            return (img_id, None, "embedding=None")
        f = f.astype("float32").ravel()
        return (img_id, f, None)
    except Exception as e:
        return (img_id, None, str(e))

# ---------------- HDF5 Writer ---------------- #

class H5ExtWriter:
    def __init__(self, out_path: str, feat_dim: int, chunk_size: int = 512):
        self.f = h5py.File(out_path, "w")
        self.ds_features = self.f.create_dataset(
            "features", shape=(0, feat_dim), maxshape=(None, feat_dim),
            dtype="float32", chunks=(min(chunk_size, 1024), feat_dim), compression="lzf"
        )
        self.ds_ids = self.f.create_dataset(
            "ids", shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"), chunks=True
        )
        self.count = 0
        self.feat_dim = feat_dim

    def append_batch(self, ids: List[str], feats):
        feats = np.asarray(feats, dtype="float32")
        n = feats.shape[0]
        new_total = self.count + n
        self.ds_features.resize((new_total, self.feat_dim))
        self.ds_features[self.count:new_total, :] = feats
        self.ds_ids.resize((new_total,))
        self.ds_ids[self.count:new_total] = ids
        self.count = new_total

    def close(self):
        self.f.flush(); self.f.close()

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--num-workers", type=int, default=max(1,(cpu_count() or 2)-1))
    p.add_argument("--batch-write", type=int, default=512)
    args = p.parse_args()

    images = sorted(glob.glob(os.path.join(args.images_dir, "**", args.pattern), recursive=True))
    images = [p for p in images if os.path.isfile(p)]
    if not images:
        print("⚠️ Aucune image trouvée."); sys.exit(1)

    # Probe dimension
    first = get_image_embedding(images[0])
    feat_dim = first.shape[0]
    print(f"✅ Dim features = {feat_dim}")

    writer = H5ExtWriter(args.output, feat_dim=feat_dim, chunk_size=args.batch_write)

    worker = partial(_process_one, root=args.images_dir, include_x=False)
    t0 = time.time(); ok=0; err=0; buf_ids=[]; buf_feats=[]
    with Pool(processes=args.num_workers) as pool:
        for img_id, feat, e in pool.imap_unordered(worker, images, chunksize=16):
            if feat is not None:
                buf_ids.append(img_id); buf_feats.append(feat); ok+=1
                if len(buf_ids)>=args.batch_write:
                    writer.append_batch(buf_ids, buf_feats); buf_ids=[]; buf_feats=[]
            else:
                err+=1
                print(f"⚠️ Fail {img_id}: {e}")
        if buf_ids:
            writer.append_batch(buf_ids, buf_feats)
    writer.close()
    dt=time.time()-t0
    print(f"✨ Fini. {ok}/{len(images)} encodées en {dt:.1f}s ({ok/dt:.2f} img/s). Erreurs={err}. → {args.output}")

if __name__=="__main__":
    main()

