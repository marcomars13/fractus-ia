import os, sys, csv, time, json, argparse, hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

FLICKR_ENDPOINT = "https://www.flickr.com/services/rest/"

EXTRAS = ",".join([
    "geo","date_taken","owner_name","license",
    "url_o","url_k","url_h","url_l","url_c","url_z","url_m"
])

URL_KEYS = ["url_o","url_k","url_h","url_l","url_c","url_z","url_m"]

def api_get(params):
    params = {
        **params,
        "format": "json",
        "nojsoncallback": 1,
    }
    r = requests.get(FLICKR_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_best_url(photo):
    for k in URL_KEYS:
        if k in photo: 
            return photo[k]
    return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("FLICKR_API_KEY"), required=False)
    ap.add_argument("--out", default="data/flickr", help="Dossier racine dataset Flickr")
    ap.add_argument("--per-page", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=999999)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--target-gb", type=float, default=20.0, help="Taille visée en Go")
    ap.add_argument("--bbox", default=None, help="min_lon,min_lat,max_lon,max_lat (optionnel)")
    ap.add_argument("--licenses", default="1,2,3,4,5,6,7,8,9,10", help="IDs licences Flickr (public)")
    ap.add_argument("--min-upload-date", default=None)
    ap.add_argument("--safe-search", type=int, default=1)
    args = ap.parse_args()

    if not args.api_key:
        print("❌ Manque --api-key ou FLICKR_API_KEY", file=sys.stderr)
        sys.exit(1)

    root = ensure_dir(args.out)
    img_dir = ensure_dir(os.path.join(root, "images"))
    meta_csv = os.path.join(root, "flickr_meta.csv")
    ckpt_path = os.path.join(root, "checkpoint.json")
    seen_path = os.path.join(root, "seen_ids.txt")

    seen = set()
    if os.path.exists(seen_path):
        with open(seen_path, "r") as f:
            for line in f: 
                seen.add(line.strip())

    downloaded_bytes = 0
    start_page = 1
    if os.path.exists(ckpt_path):
        try:
            ck = json.load(open(ckpt_path))
            start_page = ck.get("page", 1)
            downloaded_bytes = ck.get("downloaded_bytes", 0)
            print(f"🔁 Reprise depuis page={start_page}, déjà {human(downloaded_bytes)} téléchargés.")
        except Exception:
            pass

    meta_lock = threading.Lock()
    size_lock = threading.Lock()

    meta_file_exists = os.path.exists(meta_csv)
    meta_f = open(meta_csv, "a", newline="")
    writer = csv.writer(meta_f)
    if not meta_file_exists:
        writer.writerow(["id","filename","lat","lon","datetaken","owner","license","best_url","width","height","page_hint"])

    target_bytes = int(args.target_gb * (1024**3))
    page = start_page
    total_new = 0

    try:
        while page <= args.max_pages and downloaded_bytes < target_bytes:
            params = {
                "method": "flickr.photos.search",
                "api_key": args.api_key,
                "extras": EXTRAS,
                "per_page": args.per_page,
                "page": page,
                "has_geo": 1,
                "content_type": 1,
                "safe_search": args.safe_search,
                "accuracy": 11,
                "license": args.licenses,
                "sort": "date-posted-desc",
            }
            if args.bbox:
                params["bbox"] = args.bbox
            if args.min_upload_date:
                params["min_upload_date"] = args.min_upload_date

            data = api_get(params)
            photos = data.get("photos", {})
            items = photos.get("photo", [])

            if not items:
                print(f"⚠️ Page {page}: vide. On continue.")
                page += 1
                time.sleep(0.5)
                continue

            def download_one(p):
                pid = str(p["id"])
                if pid in seen:
                    return 0, None
                url = pick_best_url(p)
                if not url:
                    return 0, None
                try:
                    r = requests.get(url, stream=True, timeout=60)
                    r.raise_for_status()
                    ctype = r.headers.get("Content-Type","image/jpeg")
                    ext = ".jpg"
                    if "png" in ctype: 
                        ext = ".png"
                    out_name = f"{pid}{ext}"
                    out_path = os.path.join(img_dir, out_name)
                    with open(out_path, "wb") as w:
                        for chunk in r.iter_content(chunk_size=1024*128):
                            if chunk: w.write(chunk)
                    size = os.path.getsize(out_path)
                    with meta_lock:
                        writer.writerow([
                            pid, out_name,
                            p.get("latitude",""), p.get("longitude",""),
                            p.get("datetaken",""), p.get("ownername",""),
                            p.get("license",""), url,
                            p.get("width_o","") or p.get("width_k","") or p.get("width_h","") or p.get("width_l","") or "",
                            p.get("height_o","") or p.get("height_k","") or p.get("height_h","") or p.get("height_l","") or "",
                            page
                        ])
                        meta_f.flush()
                        with open(seen_path, "a") as sf:
                            sf.write(pid + "\n")
                        seen.add(pid)
                    return size, out_name
                except Exception:
                    return 0, None

            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futures = [ex.submit(download_one, p) for p in items]
                for fut in as_completed(futures):
                    sz, fname = fut.result()
                    if sz > 0:
                        with size_lock:
                            downloaded_bytes += sz
                            total_new += 1

            json.dump({"page": page+1, "downloaded_bytes": downloaded_bytes}, open(ckpt_path,"w"))
            print(f"📦 Page {page} | +{total_new} fichiers | total={human(downloaded_bytes)} / cible={human(target_bytes)}")
            total_new = 0

            if downloaded_bytes >= target_bytes:
                print("✅ Cible atteinte.")
                break

            page += 1
            time.sleep(0.3)

    finally:
        meta_f.close()

if __name__ == "__main__":
    main()

