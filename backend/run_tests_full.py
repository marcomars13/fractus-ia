"""
run_tests_full.py — Comparaison Plonk vs Plonk+Fractus
"""

import sys
import cv2
import numpy as np
from PIL import Image
import traceback
import plonk_infer  # Ton wrapper Plonk
from fractus import compute_fractus_scores
from fractus_profile_utils import load_fractus_profile

# ------------------------
# 🔹 Plonk seul
# ------------------------
def run_plonk_api(image_arr):
    print("🚀 [run_plonk_api] start")
    return plonk_infer.plonk_predict(image_arr)  # ✅ correction ici


# ------------------------
# 🔹 Plonk + Fractus Full
# ------------------------
def run_fractus_full_api(image_arr, profile):
    print("🚀 [run_fractus_full_api] start")

    # Prediction Plonk
    plonk_res = plonk_infer.plonk_predict(image_arr)
    lat = plonk_res[0]["lat"]
    lon = plonk_res[0]["lon"]

    # Score fractus
    scores = compute_fractus_scores(image_arr)

    # Ajustement avec params
    params = profile["params"]
    lat += params.get("alpha_mean", 0.0) * float(np.mean(scores))
    lon += params.get("alpha_std", 0.0) * float(np.std(scores))

    # Clamp pour éviter les valeurs hors limites
    lat = max(-90.0, min(90.0, lat))
    lon = max(-180.0, min(180.0, lon))

    return {"lat": lat, "lon": lon, "meta": {"engine": "Plonk+Fractus"}}


# ------------------------
# 🔹 Script principal
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python run_tests_full.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        image_arr = np.array(Image.open(image_path).convert("RGB"))

        # Chargement du profil
        profile = load_fractus_profile()

        # Exécuter Plonk seul
        res_plonk = run_plonk_api(image_arr)

        # Exécuter Plonk + Fractus
        res_fractus = run_fractus_full_api(image_arr, profile)

        # Comparaison
        print("\n📊 Résultats comparés :")
        print(f"   ➤ Plonk seul       : lat={res_plonk[0]['lat']}, lon={res_plonk[0]['lon']}")
        print(f"   ➤ Plonk + Fractus  : lat={res_fractus['lat']}, lon={res_fractus['lon']}  [Plonk+Fractus]")

        # Distance approx
        dist = np.sqrt((res_plonk[0]['lat'] - res_fractus['lat'])**2 +
                       (res_plonk[0]['lon'] - res_fractus['lon'])**2) * 111
        print(f"📏 Distance (ajustement) : {dist:.2f} km")

    except Exception as e:
        print("❌ Erreur pendant le test :", e)
        traceback.print_exc()

