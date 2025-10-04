#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test local API (Plonk vs Plonk+Fractus full)

Usage:
    python test_local.py --image /chemin/vers/test.jpg \
                         --lat 48.8566 --lon 2.3522
"""

import requests
import argparse
import math
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en kilomètres entre deux points (lat/lon en degrés)."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def post_image(endpoint, image_path):
    url = f"{BASE_URL}{endpoint}"
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(url, files=files)
    r.raise_for_status()
    return r.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Chemin de l'image à tester")
    parser.add_argument("--lat", type=float, help="Latitude ground truth")
    parser.add_argument("--lon", type=float, help="Longitude ground truth")
    args = parser.parse_args()

    print("📤 Test API local sur:", args.image)

    # Plonk seul
    plonk_res = post_image("/infer/plonk", args.image)
    print("Plonk →", plonk_res)

    # Plonk+Fractus full
    fractus_res = post_image("/infer/plonk_fractus", args.image)
    print("Plonk+Fractus →", fractus_res)

    # Comparaison avec GT si dispo
    if args.lat is not None and args.lon is not None:
        gt_lat, gt_lon = args.lat, args.lon
        d_plonk = haversine_km(gt_lat, gt_lon, plonk_res.get("lat"), plonk_res.get("lon"))
        d_fractus = haversine_km(gt_lat, gt_lon, fractus_res.get("lat"), fractus_res.get("lon"))

        print(f"\n🎯 Ground Truth: ({gt_lat}, {gt_lon})")
        print(f"📏 Distance Plonk: {d_plonk:.3f} km")
        print(f"📏 Distance Plonk+Fractus: {d_fractus:.3f} km")

        if d_plonk and d_fractus:
            delta = d_plonk - d_fractus
            signe = "↓" if delta > 0 else "↑"
            print(f"✅ Gain Fractus: {signe} {abs(delta):.3f} km")
    else:
        print("\n⚠️ Pas de GT fournie → distances non calculées.")

if __name__ == "__main__":
    main()

