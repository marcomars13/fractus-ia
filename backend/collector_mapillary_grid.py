import os
import json
import csv
import time
import pathlib
import requests
import numpy as np

# ============================
# CONFIG
# ============================
ACCESS_TOKEN = "MLY|24505315075792684|b49dc3aada8a840a2988c32972dae35c"

# France métropolitaine
MIN_LON, MIN_LAT = -5.5, 41.0
MAX_LON, MAX_LAT = 9.7, 51.5

GRID_SIZE = 5  # grille 5x5 -> 25 zones
LIMIT = 1000   # max par appel
FIELDS = "id,thumb_1024_url,computed_geometry,sequence_id,captured_at,compass_angle"

OUT_DIR = pathlib.Path("mapillary_france_grid")
CSV_PATH = OUT_DIR / "images.csv"
JSONL_PATH = OUT_DIR / "images.jsonl"

SLEEP_BETWEEN_CALLS = 0.3

# ============================
# UTILS
# ============================
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
# GRID GENERATOR
# ============================
def generate_grid(min_lon, min_lat, max_lon, max_lat, grid_size):
    lons = np.linspace(min_lon, max_lon, grid_size + 1)
    lats = np.linspace(min_lat, max_lat, grid_size + 1)
    zones = []
    for i in range(grid_size):
        for j in range(grid_size):
            zones.append((
                lons[i], lats[j], lons[i+1], lats[j+1], f"zone_{i}_{j}"
            ))
    return zones

# ============================
# COLLECTOR
# ============================
def collect_zone(bbox, zone_id, csv_writer, jf):
    base_url = "https://graph.mapillary.com/images"
    params = {
        "fields": FIELDS,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": str(LIMIT),
    }

    total = 0
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

            csv_writer.writerow([_id, lon, lat, thumb, seq, cap, ang, zone_id])
            jf.write(json.dumps(item, ensure_ascii=False) + "\n")

            total += 1

        print(f"[INFO] Zone {zone_id} : {total} images...")

        next_url = extract_next_paging_url(obj)
        if not next_url:
            break

        obj = get_page(next_url)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return total

# ============================
# MAIN
# ============================
def main():
    ensure_dirs()

    zones = generate_grid(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, GRID_SIZE)
    total_global = 0

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["id","lon","lat","thumb_1024_url","sequence_id","captured_at","compass_angle","zone_id"])

    with open(JSONL_PATH, "w", encoding="utf-8") as jf:
        for (minlon, minlat, maxlon, maxlat, zone_id) in zones:
            print(f"\n=== Collecte {zone_id} ({minlon},{minlat},{maxlon},{maxlat}) ===")
            try:
                total = collect_zone((minlon, minlat, maxlon, maxlat), zone_id, csv_writer, jf)
                total_global += total
            except Exception as e:
                print(f"[ERREUR] Zone {zone_id}: {e}")
            time.sleep(1)  # pause entre zones

    csv_file.close()
    print(f"\n[OK] Collecte terminée. Total: {total_global} images.\n- CSV: {CSV_PATH}\n- JSONL: {JSONL_PATH}")

if __name__ == "__main__":
    main()

