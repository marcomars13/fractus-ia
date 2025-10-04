#!/usr/bin/env python3
import requests
import argparse
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def infer_plonk(img_path: Path):
    url = f"{API_URL}/infer/plonk"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    return resp.json()

def infer_fractus(img_path: Path):
    url = f"{API_URL}/infer/plonk_fractus"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    return resp.json()

def main():
    parser = argparse.ArgumentParser(description="Test endpoints Plonk & Fractus")
    parser.add_argument("--img", required=True, help="Chemin de l’image à tester")
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f"❌ Image introuvable: {img_path}")
        return

    print(f"➡️ Test de l'image : {img_path}")

    # Test Plonk
    try:
        plonk_res = infer_plonk(img_path)
        print("\n📍 Résultat Plonk :")
        print(plonk_res)
    except Exception as e:
        print(f"❌ Erreur Plonk: {e}")

    # Test Fractus
    try:
        fractus_res = infer_fractus(img_path)
        print("\n🌀 Résultat Fractus :")
        print(fractus_res)
    except Exception as e:
        print(f"❌ Erreur Fractus: {e}")

if __name__ == "__main__":
    main()

