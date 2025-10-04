import numpy as np
from PIL import Image
from fractus_core import run_fractus_full_api, extract_vector

def run_fractus_ultimate(img_path: str):
    """
    Pipeline complet Fractus : prédiction lat/lon via run_fractus_full_api.
    """
    try:
        pil_img = Image.open(img_path).convert("RGB")
        result = run_fractus_full_api(pil_img)

        if "lat" in result and "lon" in result:
            return {"lat": result["lat"], "lon": result["lon"]}
        else:
            print(f"⚠️ Résultat inattendu: {result}")
            return {"lat": None, "lon": None}
    except Exception as e:
        print(f"⚠️ Erreur run_fractus_ultimate sur {img_path}: {e}")
        return {"lat": None, "lon": None}

