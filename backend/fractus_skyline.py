import numpy as np
import cv2
from skyline_extractor import extract_skyline_signature
from fractus_core import run_fractus_full_api

def run_fractus_skyline(image_path: str):
    """
    Convertit la skyline en image 2D (profil tracé) pour run_fractus_full_api.
    """
    try:
        skyline_vec = extract_skyline_signature(image_path)

        # Créer une image 2D noire
        h, w = 64, len(skyline_vec)
        img = np.zeros((h, w), dtype=np.float32)

        # Tracer la skyline (normalisée dans [0,h-1])
        y_coords = (skyline_vec * (h - 1)).astype(int)
        for x, y in enumerate(y_coords):
            img[y, x] = 1.0  # tracer un point blanc

        # Redimensionner en 64x64
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Ajouter canal (64,64,1)
        arr = np.expand_dims(img_resized, axis=-1)

        # Appel Fractus
        fractus_out = run_fractus_full_api(arr)

        print(f"🔎 Retour brut Fractus: {fractus_out}")

        if isinstance(fractus_out, dict) and "lat" in fractus_out and "lon" in fractus_out:
            return {
                "lat": float(fractus_out["lat"]),
                "lon": float(fractus_out["lon"]),
                "score": float(fractus_out.get("score", 0.0))
            }
        else:
            return {"lat": None, "lon": None, "error": f"Format inattendu: {fractus_out}"}

    except Exception as e:
        return {"lat": None, "lon": None, "error": str(e)}

if __name__ == "__main__":
    test_img = "/Users/marco/mapillary_france_adaptive/thumbs/1000091355214658.jpg"
    out = run_fractus_skyline(test_img)
    print("✅ Test Fractus Skyline:", out)

