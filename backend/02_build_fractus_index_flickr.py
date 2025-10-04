import os, sys, csv, json, argparse, hashlib
import numpy as np

USE_SK = True
try:
    from sklearn.neighbors import KDTree as SKKDTree
except Exception:
    USE_SK = False
    from scipy.spatial import cKDTree as CKDTree

from PIL import Image
try:
    from fractus_core import extract_vector
except Exception:
    print("❌ Impossible d'importer fractus_core.extract_vector", file=sys.stderr)
    sys.exit(1)

def stable_split(photo_id, modulo=10):
    h = hashlib.md5(photo_id.encode("utf-8")).hexdigest()
    return int(h, 16) % modulo

def read_meta(meta_csv, images_dir):
    rows = []
    with open(meta_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row["filename"]
            lat, lon = row["lat"], row["lon"]
            if not fn or not lat or not lon:
                continue
            imgp = os.path.join(images_dir, fn)
            if os.path.exists(imgp):
                rows.append({
                    "id": row["id"],
                    "filename": fn,
                    "path": imgp,
                    "lat": float(lat),
                    "lon": float(lon)
                })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/flickr/flickr_meta.csv")
    ap.add_argument("--images-dir", default="data/flickr/images")
    ap.add_argument("--out-dir", default="data/flickr/index")
    ap.add_argument("--modulo", type=int, default=10, help="split 90/10 si 10")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = pas de limite")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_meta(args.meta, args.images_dir)
    if not rows:
        print("❌ Aucune entrée valide")
        sys.exit(1)
    print(f"📄 {len(rows)} images valides avec lat/lon.")

    train, test = [], []
    for r in rows:
        if stable_split(r["id"], args.modulo) == 0:
            test.append(r)
        else:
            train.append(r)

    if args.limit-train and args.limit-train < len(train):
        train = train[:args.limit-train]

    def write_split(lst, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename","lat","lon"])
            for r in lst:
                w.writerow([r["filename"], r["lat"], r["lon"]])

    train_csv = os.path.join(args.out_dir, "flickr_train.csv")
    test_csv  = os.path.join(args.out_dir, "flickr_test.csv")
    write_split(train, train_csv)
    write_split(test, test_csv)
    print(f"✂️  Split → train={len(train)} | test={len(test)}")

    vecs, coords, ids = [], [], []
    for i, r in enumerate(train, 1):
        try:
            v = extract_vector(r["path"])
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            vecs.append(v)
            coords.append([r["lat"], r["lon"]])
            ids.append(r["filename"])
        except Exception:
            continue
        if i % 1000 == 0:
            print(f"   • {i}/{len(train)} vecteurs extraits")

    V = np.stack(vecs, axis=0)
    C = np.asarray(coords, dtype=np.float32)
    I = np.array(ids, dtype=object)

    np.save(os.path.join(args.out_dir, "vectors.npy"), V)
    np.save(os.path.join(args.out_dir, "coords.npy"),  C)
    np.save(os.path.join(args.out_dir, "ids.npy"),     I)

    if USE_SK:
        tree = SKKDTree(V, leaf_size=40)
    else:
        tree = CKDTree(V)

    import pickle
    with open(os.path.join(args.out_dir, "kdtree.pkl"), "wb") as f:
        pickle.dump({"type": "sk" if USE_SK else "ckd", "tree": tree}, f)

    meta = {
        "train": len(train),
        "test": len(test),
        "vector_dim": int(V.shape[1]),
        "index_type": "sklearn" if USE_SK else "scipy_ckdtree"
    }
    json.dump(meta, open(os.path.join(args.out_dir,"index_meta.json"),"w"), indent=2)
    print(f"✅ Index construit : {V.shape} → {os.path.join(args.out_dir,'kdtree.pkl')}")
    print(f"📄 Splits : {train_csv} | {test_csv}")

if __name__ == "__main__":
    main()

