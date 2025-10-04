import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ✅ Chargement global du modèle pour éviter de le recharger à chaque appel
_plonk_model = None
_plonk_processor = None


def load_plonk_model():
    """
    Charge le modèle CLIP utilisé comme proxy de Plonk.
    ⚠️ Remplace 'openai/clip-vit-base-patch32' par ton vrai modèle si besoin.
    """
    global _plonk_model, _plonk_processor
    if _plonk_model is None or _plonk_processor is None:
        print("🚀 Chargement du modèle Plonk (CLIP)...")
        _plonk_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _plonk_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Modèle Plonk chargé")
    return _plonk_model, _plonk_processor


def plonk_predict(image: Image.Image):
    """
    Prend une image PIL en entrée et renvoie une prédiction factice
    (à adapter selon ton vrai modèle Plonk).
    Ici, on renvoie juste des coordonnées neutres pour tester le flux.
    """
    model, processor = load_plonk_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # ⚠️ Placeholder simplifié : à remplacer par vraie logique Plonk
    lat = float(outputs[0][0]) % 90    # valeur pseudo-GPS
    lon = float(outputs[0][1]) % 180   # valeur pseudo-GPS

    return {"lat": lat, "lon": lon, "raw": outputs.tolist()}


def run_plonk_api(image_path: str):
    """
    Wrapper de compatibilité pour compare_plonk_fractus_ultimate.py :
    - Charge l'image depuis son chemin
    - Appelle plonk_predict()
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return plonk_predict(img)

