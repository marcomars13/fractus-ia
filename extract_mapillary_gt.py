#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, json, os
from pathlib import Path

# === PARAMÈTRES (chemins corrigés) ===
MAPILLARY_CSV = "/Users/marco/mapillary_france_adaptive/images.csv"
MAPILLARY_JSONL = "/Users/marco/mapillary_france_adaptive/images.jsonl"
OUT_CSV = "data/ground_truth.csv"   # fichier de sortie

def extract_from_csv(path, out_path):
    with open(path, newline='', encoding="utf-8") as f, open(out_path, "w", newline="", encoding="utf-8") as out:
        rd = csv.DictReader(f)
        wr = csv.writer(out)
        wr.writerow(["filename", "lat", "lon"])
        for row in rd:
            fname = row.get("filename") or row.get("file") or row.get("id")
            if not fname: 
                continue
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if not lat or not lon: 
                continue
            wr.writerow([Path(fname).name, lat, lon])

def extract_from_jsonl(path, out_path):
    with open(path, "r", encoding="utf-8") as f, open(out_path, "w", newline="", encoding="utf-8") as out:
        wr = csv.writer(out)
        wr.writerow(["filename", "lat", "lon"])
        for line in f:
            try:
                row = json.loads(line)
                lat = row.get("lat") or row.get("latitude")
                lon = row.get("lon") or row.get("longitude")
                img_id = row.get("filename") or row.get("file") or row.get("id")
                if not img_id or not lat or not lon: 
                    continue
                fname = Path(img_id).name
                wr.writerow([fname, lat, lon])
            except json.JSONDecodeError:
                continue

def main():
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(MAPILLARY_CSV):
        print(f"➡️ Extraction depuis CSV: {MAPILLARY_CSV}")
        extract_from_csv(MAPILLARY_CSV, out_path)
    elif os.path.exists(MAPILLARY_JSONL):
        print(f"➡️ Extraction depuis JSONL: {MAPILLARY_JSONL}")
        extract_from_jsonl(MAPILLARY_JSONL, out_path)
    else:
        print("❌ Aucun fichier Mapillary trouvé (CSV ou JSONL)")

    print(f"✅ Fichier ground truth généré: {OUT_CSV}")

if __name__ == "__main__":
    main()

