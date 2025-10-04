# plonk_model.py
# Wrapper pour interroger ton API Plonk en local

import requests

def predict_plonk(img_path: str):
    """
    Envoie une image au backend Plonk (FastAPI) et récupère la prédiction (lat, lon).
    """
    url = "http://127.0.0.1:8000/infer/plonk"
    with open(img_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()

    # ⚠️ ton backend renvoie une liste [{'lat': ..., 'lon': ...}]
    if isinstance(data, list) and len(data) > 0:
        return data[0]["lat"], data[0]["lon"]
    # fallback
    return None, None

