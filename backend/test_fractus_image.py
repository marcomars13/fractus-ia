import os
import cv2
import numpy as np
from fractus_core import run_fractus_full_api

# Dossier d'images Mapillary
IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"

def run_fractus_image(image_path: str):
    """
    Teste Fractus avec une image brute (RGB normalisée).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"❌ Impossible de lire {image_path}")

        # Conversion en float32 et normalisation
        arr = img.astype(np.float32) / 255.0

        # Appel Fractus
        out = run_fractus_full_api(arr)

        print(f"🔎 Image: {os.path.basename(image_path)}")
        print(f"   ↳ Retour brut: {out}")

        return out

    except Exception as e:
        return {"lat": None, "lon": None, "error": str(e)}

def main():
    # Prend la première image dispo
    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if not images:
        raise FileNotFoundError(f"❌ Aucune image trouvée dans {IMG_DIR}")

    test_img = os.path.join(IMG_DIR, images[0])
    print(f"🖼️ Test sur {test_img}")

    out = run_fractus_image(test_img)

    if "lat" in out and "lon" in out:
        print(f"✅ Résultat Fractus Image : lat={out['lat']}, lon={out['lon']}, score={out.get('score')}")
    else:
        print(f"⚠️ Format inattendu: {out}")

if __name__ == "__main__":
    main()

