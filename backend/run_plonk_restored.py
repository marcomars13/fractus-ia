"""
run_plonk_restored.py
Wrapper robuste pour utiliser un modèle Plonk restauré localement (backend/fractus_restored_model.bin)
- Tente plusieurs loaders : torch, pickle, joblib, numpy
- Si le modèle a une méthode prédictive (predict / infer / __call__ / forward), elle est utilisée
- Sinon, retourne un fallback déterministe basé sur le hash du nom de fichier (lat, lon)
Usage:
    from run_plonk_restored import run_plonk_restored
    out = run_plonk_restored("/path/to/img.jpg")
    -> out is dict { 'lat':..., 'lon':..., 'confidence':..., 'error': Optional[str] }
"""
import os
import sys
import hashlib
import json
import traceback

# chemins par défaut
DEFAULT_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "fractus_restored_model.bin"),
    os.path.join(os.path.dirname(__file__), "fractus_restored_model.pt"),
    os.path.join(os.path.dirname(__file__), "plonk_restored.bin"),
]

def _hash_to_latlon(name: str):
    """Fallback déterministe: transforme un nom en lat/lon plausibles."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    # prends 16 premiers chars pour créer deux entiers
    a = int(h[0:16], 16)
    b = int(h[16:32], 16)
    # map a -> lat [-90,90], b -> lon [-180,180]
    lat = (a % (1800000)) / 10000.0 - 90.0   # précision décimale raisonnable
    lon = (b % (3600000)) / 10000.0 - 180.0
    confidence = 0.01 + (int(h[32:40], 16) % 90) / 100.0  # 0.01..0.90 pseudo-confiance
    return {"lat": round(lat, 6), "lon": round(lon, 6), "confidence": round(confidence, 3)}

def _try_load_with_torch(model_path):
    try:
        import torch
        print(f"🧪 [torch] Tentative chargement {model_path}")
        model = torch.load(model_path, map_location="cpu")
        print("✅ [torch] fichier chargé.")
        return model
    except Exception as e:
        print(f"⚠️ [torch] échec: {e}")
        return None

def _try_load_with_pickle(model_path):
    try:
        import pickle
        print(f"🧪 [pickle] Tentative chargement {model_path}")
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        print("✅ [pickle] fichier chargé.")
        return obj
    except Exception as e:
        print(f"⚠️ [pickle] échec: {e}")
        return None

def _try_load_with_joblib(model_path):
    try:
        import joblib
        print(f"🧪 [joblib] Tentative chargement {model_path}")
        obj = joblib.load(model_path)
        print("✅ [joblib] fichier chargé.")
        return obj
    except Exception as e:
        print(f"⚠️ [joblib] échec: {e}")
        return None

def _try_load_with_numpy(model_path):
    try:
        import numpy as np
        print(f"🧪 [numpy] Tentative chargement {model_path}")
        arr = np.load(model_path, allow_pickle=True)
        print("✅ [numpy] fichier chargé.")
        return arr
    except Exception as e:
        print(f"⚠️ [numpy] échec: {e}")
        return None

def _find_model_path(user_path=None):
    # priorité : chemin fourni, sinon chemins par défaut
    if user_path and os.path.exists(user_path):
        return user_path
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None

def _call_model_predict(model, img_path):
    """
    Tente les méthodes prédictives usuelles sur l'objet model.
    Doit retourner un dict {'lat': float, 'lon': float, 'confidence': float}
    """
    # 1) méthode predict(img_path) ou predict(image_array)
    for meth in ("predict", "infer", "__call__", "forward", "predict_one"):
        fn = getattr(model, meth, None)
        if callable(fn):
            try:
                print(f"🔎 Appel de la méthode modèle '{meth}'")
                out = fn(img_path)
                # si la méthode attend une image array, certains wrappers permettent path->auto
                # on accepte dicts ou listes contenant dicts
                if isinstance(out, dict) and "lat" in out and "lon" in out:
                    return out
                if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], dict):
                    return out[0]
                # sinon, si out est une structure simple, essaye d'interpréter
                print(f"⚠️ Méthode '{meth}' a renvoyé (non standard): {type(out)} => {out}")
            except Exception as e:
                print(f"⚠️ Erreur en appelant '{meth}': {e}")
                traceback.print_exc()
    # 2) si model est dict-like contenant 'predict' key
    try:
        if isinstance(model, dict):
            for k in ("predict", "infer"):
                if k in model and callable(model[k]):
                    try:
                        out = model[k](img_path)
                        if isinstance(out, dict) and "lat" in out and "lon" in out:
                            return out
                    except Exception:
                        pass
    except Exception:
        pass
    return None

# Mémoire du modèle chargé (singleton)
_LOADED_MODEL = None
_LOADED_MODEL_PATH = None

def _load_model(user_model_path=None):
    global _LOADED_MODEL, _LOADED_MODEL_PATH
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    model_path = _find_model_path(user_model_path)
    if model_path is None:
        print("⚠️ Aucun fichier de modèle trouvé sur les chemins connus.")
        return None

    print(f"🔁 Tentative de chargement du modèle depuis : {model_path}")
    loaders = [
        _try_load_with_torch,
        _try_load_with_joblib,
        _try_load_with_pickle,
        _try_load_with_numpy,
    ]
    for loader in loaders:
        try:
            obj = loader(model_path)
            if obj is not None:
                _LOADED_MODEL = obj
                _LOADED_MODEL_PATH = model_path
                print(f"✅ Modèle chargé via {loader.__name__}")
                return _LOADED_MODEL
        except Exception as e:
            print(f"⚠️ Loader {loader.__name__} a échoué : {e}")
    print("❌ Aucun loader n'a pu charger le modèle correctement.")
    return None

def run_plonk_restored(img_path: str, model_path: str = None):
    """
    API principale : renvoie dict {'lat','lon','confidence', 'error' (optionnel)}
    - img_path : chemin vers l'image
    - model_path : chemin explicite du modèle restauré (optionnel)
    """
    # 1) vérifications basiques
    if not os.path.exists(img_path):
        return {"lat": 0.0, "lon": 0.0, "confidence": 0.0, "error": f"Image not found: {img_path}"}

    model = _load_model(model_path)
    if model is None:
        # fallback déterministe basé sur nom du fichier
        res = _hash_to_latlon(os.path.basename(img_path))
        res["error"] = "Model not loadable - using deterministic fallback"
        print("⚠️ Utilisation du fallback déterministe (model absent ou illisible).")
        return res

    # 2) tenter l'appel prédictif du modèle
    try:
        out = _call_model_predict(model, img_path)
        if out and isinstance(out, dict) and "lat" in out and "lon" in out:
            # Normalise et renvoie
            out.setdefault("confidence", float(out.get("confidence", 1.0)))
            return {"lat": float(out["lat"]), "lon": float(out["lon"]), "confidence": float(out["confidence"])}
        else:
            print("⚠️ Le modèle a été chargé mais n'a pas renvoyé de prédiction exploitable.")
            # on essaye un dernier recours : si model est un numpy array avec shape->on génère fallback
            res = _hash_to_latlon(os.path.basename(img_path))
            res["error"] = "Model loaded but no usable predict method - using deterministic fallback"
            return res
    except Exception as e:
        print(f"❌ Exception lors de la prédiction: {e}")
        traceback.print_exc()
        res = _hash_to_latlon(os.path.basename(img_path))
        res["error"] = f"Exception during predict: {e} - using deterministic fallback"
        return res

# Petit test si script exécuté directement
if __name__ == "__main__":
    import sys
    test_img = None
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        # fallback image: première du dossier mapillary si disponible
        cand_dir = os.path.join(os.path.expanduser("~"), "mapillary_france_adaptive", "thumbs")
        if os.path.isdir(cand_dir):
            files = [f for f in os.listdir(cand_dir) if f.endswith(".jpg")]
            if files:
                test_img = os.path.join(cand_dir, files[0])
    if not test_img:
        print("Usage: python run_plonk_restored.py /path/to/image.jpg")
        sys.exit(1)

    print("🧾 Test run_plonk_restored sur:", test_img)
    out = run_plonk_restored(test_img)
    print("→ Résultat :", out)

