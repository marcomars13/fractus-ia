import os
import json
import csv
import time
import pathlib
import requests

# ============================
# CONFIG
# ============================
ACCESS_TOKEN = "MLY|24505315075792684|b49dc3aada8a840a2988c32972dae35c"

# France métropolitaine
BBOX = "-5.5,41.0,9.7,51.5"  # minLon, minLat, maxLon, maxLat
LIMIT = 1000  # max par appel
FIELDS = "id,thumb_1024_url,computed_geometry,sequence_id,captured_at,compass_angle"

OUT_DIR = pathlib.Path("mapillary_france")
CSV_PATH = OUT_DIR / "images.csv"
JSONL_PATH = OUT_DIR / "images.jsonl"
DOWNLOAD_THUMBS = False  # ⚠️ désactivé car la France entière = des milliers d’images
SLEEP_BETWEEN_CALLS = 0.3

# ============================
# UTILS
# ============================
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DOWNLOAD_THUMBS:
        (OUT_DIR / "thumbs").mkdir(parents=True, exist_ok=True)

def get_page(url, params=None):
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

def extract_next_paging_url(obj):
    paging = obj.get("paging")
    if not paging:
        return None
    return paging.get("next")

# ============================
# MAIN
# ============================
def main():
    ensure_dirs()

    base_url = "https://graph.mapillary.com/images"
    params = {
        "fields": FIELDS,
        "bbox": BBOX,
        "limit": str(LIMIT),
    }

    total = 0
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["id","lon","lat","thumb_1024_url","sequence_id","captured_at","compass_angle"])

    with open(JSONL_PATH, "w", encoding="utf-8") as jf:
        obj = get_page(base_url, params)

        while True:
            data = obj.get("data", [])
            for item in data:
                _id = item.get("id")
                thumb = item.get("thumb_1024_url")
                seq = item.get("sequence_id")
                cap = item.get("captured_at")
                ang = item.get("compass_angle")

                lon, lat = None, None
                geom = item.get("computed_geometry") or {}
                if geom.get("type") == "Point":
                    coords = geom.get("coordinates") or []
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]

                csv_writer.writerow([_id, lon, lat, thumb, seq, cap, ang])
                jf.write(json.dumps(item, ensure_ascii=False) + "\n")

                total += 1

            print(f"[INFO] Collecté {total} images...")

            next_url = extract_next_paging_url(obj)
            if not next_url:
                break

            obj = get_page(next_url)
            time.sleep(SLEEP_BETWEEN_CALLS)

    csv_file.close()
    print(f"[OK] Collecté {total} images sur la France.\n- CSV: {CSV_PATH}\n- JSONL: {JSONL_PATH}")

if __name__ == "__main__":
    main()

