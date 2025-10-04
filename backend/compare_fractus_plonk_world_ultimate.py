import os, sys, csv, math, time, joblib, argparse, numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.neighbors import KDTree

# Imports locaux optionnels
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backend.plonk_model import run_plonk_api
except Exception:
    run_plonk_api = None  # fallback si indisponible

# ---------------- Utilities ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def must_read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ---------------- Loading ----------------
def load_fractus(fractus_file, fractus_filenames):
    obj = joblib.load(fractus_file)
    if not (isinstance(obj, tuple) and len(obj) >= 2):
        raise ValueError("❌ Fractus file must be a tuple (features, coords[, ...])")

    features, coords = obj[0], obj[1]
    if not fractus_filenames or not os.path.isfile(fractus_filenames):
        raise ValueError("❌ --fractus_filenames est requis et doit exister (mapping filename → index).")
    fnames = must_read_lines(fractus_filenames)
    if len(fnames) != features.shape[0]:
        raise ValueError(f"❌ Longueur mismatch: {len(fnames)} filenames vs features={features.shape[0]}")

    index = KDTree(features, metric="euclidean")
    return features, coords, index, fnames

def load_plonk(plonk_file):
    obj = joblib.load(plonk_file)
    if not isinstance(obj, dict) or "latlons" not in obj or "filenames" not in obj:
        raise ValueError("❌ Plonk file must be a dict with keys: 'latlons', 'filenames' (and ideally 'kdtree' & 'X').")
    latlons = np.array(obj["latlons"])
    filenames = list(obj["filenames"])
    kdtree = obj.get("kdtree", None)
    X = obj.get("X", None)  # embeddings Plonk (optionnels mais recommandés)
    return kdtree, latlons, filenames, X

# ---------------- Skyline (light) ----------------
def skyline_profile(image_path, W=128, H=128):
    try:
        from PIL import Image
        im = Image.open(image_path).convert("L").resize((W, H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        grad = np.abs(np.diff(arr, axis=0))
        prof = grad.argmax(axis=0) / float(H - 1)
        return prof.astype(np.float32)
    except Exception:
        return None

def skyline_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))

# ---------------- Fractus vote + skyline re-rank ----------------
def fractus_predict_enhanced(qvec, index_f, coords_f, topk=32,
                             skyline_q=None, skyline_ref=None, skyline_w=0.5):
    eps = 1e-6
    K = min(topk, coords_f.shape[0])
    d, ind = index_f.query(qvec.reshape(1, -1), k=K)
    d, ind = d[0], ind[0]

    # skyline re-weight (si profils ref disponibles)
    if skyline_q is not None and skyline_ref is not None:
        penalties = []
        for j in ind:
            p_ref = skyline_ref(j)
            ds = skyline_distance(skyline_q, p_ref)
            penalties.append(0.0 if ds is None else ds)
        penalties = np.array(penalties, dtype=np.float32)
        if penalties.max() > 0:
            penalties = penalties / (penalties.max() + 1e-9)
            d = d + skyline_w * penalties

    w = 1.0 / (d + eps)
    w = w / (w.sum() + eps)
    latlons = coords_f[ind]
    pred = (w.reshape(-1, 1) * latlons).sum(axis=0)
    return {"lat": float(pred[0]), "lon": float(pred[1])}

# ---------------- Plonk predict ----------------
def plonk_predict_for_image(img_name, images_dir, kdtree_p, latlons_p, filenames_p, X_plonk):
    # Prefer: utiliser l'embedding Plonk aligné au fichier (si présent)
    if X_plonk is not None:
        try:
            j = filenames_p.index(img_name)
        except ValueError:
            return None
        qvec_p = X_plonk[j]
        if kdtree_p is None:
            # S'il n'y a pas de kdtree, on peut retourner directement la GT latlon pour cette entrée,
            # mais ce serait équivalent à tricher. Mieux: retourner None → pas de prediction Plonk par KD.
            return None
        d, ind = kdtree_p.query(qvec_p.reshape(1, -1), k=1)
        nn = ind[0][0]
        return {"lat": float(latlons_p[nn, 0]), "lon": float(latlons_p[nn, 1])}

    # Sinon: fallback inference lente si API dispo
    if run_plonk_api is None:
        return None
    img_path = os.path.join(images_dir, img_name)
    try:
        pred = run_plonk_api(img_path)
        if pred and "lat" in pred and "lon" in pred:
            return {"lat": float(pred["lat"]), "lon": float(pred["lon"])}
    except Exception:
        pass
    return None

# ---------------- Worker (1 image) ----------------
def process_one(img_name, images_dir,
                features_f, coords_f, index_f, fractus_fnames,
                kdtree_p, latlons_p, filenames_p, X_plonk,
                use_skyline, topk, alpha,
                skyline_cache):
    # GT via Plonk filenames (référence)
    try:
        jgt = filenames_p.index(img_name)
        gt_lat, gt_lon = float(latlons_p[jgt, 0]), float(latlons_p[jgt, 1])
    except ValueError:
        # Image pas dans l’index Plonk → on saute
        return None

    # qvec Fractus alignée au FICHIER (indispensable)
    try:
        jf = fractus_fnames.index(img_name)
    except ValueError:
        # Pas de mapping dans Fractus → inutile de forcer (skip)
        return None
    qvec_f = features_f[jf]

    # skyline profil query (optionnel)
    skyl_q = skyline_profile(os.path.join(images_dir, img_name)) if use_skyline else None

    def skyline_ref(j):
        # profil ref basé sur le filename Fractus j → on tente de lire l'image correspondante
        try:
            ref_name = fractus_fnames[j]
            if ref_name in skyline_cache:
                return skyline_cache[ref_name]
            p = skyline_profile(os.path.join(images_dir, ref_name)) if use_skyline else None
            skyline_cache[ref_name] = p
            return p
        except Exception:
            return None

    # Fractus (vote + skyline)
    pred_f = fractus_predict_enhanced(qvec_f, index_f, coords_f,
                                      topk=topk,
                                      skyline_q=skyl_q,
                                      skyline_ref=skyline_ref,
                                      skyline_w=0.5)

    # Plonk (embedding aligné si présent; sinon API)
    pred_p = plonk_predict_for_image(img_name, images_dir, kdtree_p, latlons_p, filenames_p, X_plonk)

    # Fusion
    pred_c = {"lat": None, "lon": None}
    if pred_f and pred_f.get("lat") is not None and pred_p and pred_p.get("lat") is not None:
        lat_c = (1 - alpha) * pred_f["lat"] + alpha * pred_p["lat"]
        lon_c = (1 - alpha) * pred_f["lon"] + alpha * pred_p["lon"]
        pred_c = {"lat": lat_c, "lon": lon_c}
    elif pred_f and pred_f.get("lat") is not None:
        pred_c = pred_f
    elif pred_p and pred_p.get("lat") is not None:
        pred_c = pred_p

    # Erreurs
    err_f = haversine(gt_lat, gt_lon, pred_f["lat"], pred_f["lon"]) if pred_f.get("lat") is not None else None
    err_p = haversine(gt_lat, gt_lon, pred_p["lat"], pred_p["lon"]) if pred_p and pred_p.get("lat") is not None else None
    err_c = haversine(gt_lat, gt_lon, pred_c["lat"], pred_c["lon"]) if pred_c.get("lat") is not None else None

    return {
        "id": img_name,
        "gt_lat": gt_lat, "gt_lon": gt_lon,
        "f_lat": pred_f.get("lat"), "f_lon": pred_f.get("lon"), "f_err_km": err_f,
        "p_lat": pred_p.get("lat") if pred_p else None, "p_lon": pred_p.get("lon") if pred_p else None, "p_err_km": err_p,
        "c_lat": pred_c.get("lat"), "c_lon": pred_c.get("lon"), "c_err_km": err_c,
    }

# ---------------- Main benchmark ----------------
def benchmark(images_dir, fractus_file, plonk_file, output_csv,
             fractus_filenames,
             max_test=None, alpha=0.7, topk=32, use_skyline=True, workers=8):
    print("🚀 Benchmark Fractus vs Plonk vs Combo (Ultimate, corrigé mapping)")

    features_f, coords_f, index_f, fractus_fnames = load_fractus(fractus_file, fractus_filenames)
    print(f"📂 Fractus : features={features_f.shape}, coords={coords_f.shape}, filenames={len(fractus_fnames)}")

    kdtree_p, latlons_p, filenames_p, X_plonk = load_plonk(plonk_file)
    print(f"📂 Plonk   : latlons={latlons_p.shape}, filenames={len(filenames_p)}, "
          f"kdtree={'OK' if kdtree_p is not None else 'absent'}, X={'OK' if X_plonk is not None else 'absent'}")

    all_imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
    if max_test:
        all_imgs = all_imgs[:max_test]
        print(f"⚠️ Mode test : {len(all_imgs)} images")
    else:
        print(f"📂 {len(all_imgs)} images détectées")

    skyline_cache = {}
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [
            ex.submit(
                process_one,
                img_name, images_dir,
                features_f, coords_f, index_f, fractus_fnames,
                kdtree_p, latlons_p, filenames_p, X_plonk,
                use_skyline, topk, alpha,
                skyline_cache
            )
            for img_name in all_imgs
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Traitement (multi-coeurs)"):
            r = fut.result()
            if r is not None:  # on ignore les images non mappées
                results.append(r)
    dt = time.time() - t0
    if dt > 0:
        print(f"⏱️ {dt/60:.2f} min | {len(results)/dt:.1f} img/s")
    else:
        print("⏱️ Temps trop court pour estimer le débit.")

    if not results:
        print("❌ Aucun résultat (mapping manquant ?).")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: str(r["id"])))
    print(f"✅ Résultats → {output_csv}")

    def stats(key):
        vals = [r[key] for r in results if r[key] is not None]
        return (float(np.mean(vals)), float(np.median(vals))) if vals else (float("nan"), float("nan"))

    mf, medf = stats("f_err_km")
    mp, medp = stats("p_err_km")
    mc, medc = stats("c_err_km")

    print("\n📊 Résumé global :")
    print(f"   • Fractus : moyenne={mf:.2f} km | médiane={medf:.2f} km")
    print(f"   • Plonk   : moyenne={mp:.2f} km | médiane={medp:.2f} km")
    print(f"   • Combo   : moyenne={mc:.2f} km | médiane={medc:.2f} km")

    def top5(label):
        rows = [r for r in results if r[label] is not None]
        rows.sort(key=lambda r: r[label], reverse=True)
        return rows[:5]

    print("\n🔎 Top 5 pires erreurs (Fractus) :")
    for r in top5("f_err_km"):
        print(f"   {r['id']} | GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) → F=({r['f_lat']:.2f},{r['f_lon']:.2f}) | {r['f_err_km']:.2f} km")
    print("\n🔎 Top 5 pires erreurs (Plonk) :")
    for r in top5("p_err_km"):
        print(f"   {r['id']} | GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) → P=({r['p_lat']:.2f},{r['p_lon']:.2f}) | {r['p_err_km']:.2f} km")
    print("\n🔎 Top 5 pires erreurs (Combo) :")
    for r in top5("c_err_km"):
        print(f"   {r['id']} | GT=({r['gt_lat']:.2f},{r['gt_lon']:.2f}) → C=({r['c_lat']:.2f},{r['c_lon']:.2f}) | {r['c_err_km']:.2f} km")

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--fractus_file", required=True)
    p.add_argument("--plonk_file", required=True)
    p.add_argument("--fractus_filenames", required=True, help="TXT aligné aux features Fractus (1 filename par ligne)")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--no_skyline", action="store_true")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = p.parse_args()

    benchmark(
        images_dir=args.images_dir,
        fractus_file=args.fractus_file,
        plonk_file=args.plonk_file,
        output_csv=args.output_csv,
        fractus_filenames=args.fractus_filenames,
        max_test=args.max_test,
        alpha=args.alpha,
        topk=args.topk,
        use_skyline=not args.no_skyline,
        workers=args.workers
    )

if __name__ == "__main__":
    main()

