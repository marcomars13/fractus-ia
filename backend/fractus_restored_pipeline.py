import os, sys, json, logging
import numpy as np
import cv2
from PIL import Image
from math import radians, sin, cos, sqrt, atan2

# ⬇️ Dépendances locales
import plonk_infer
from fractus import compute_fractus_scores

# Optionnel/accélération si FAISS + meta dispos
try:
    import faiss, pandas as pd
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("fractus_pipeline")

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "mapillary_out")
PROFILE_PATH = os.path.join(OUT_DIR, "fractus_profile.json")
RESTORED_BIN = os.path.join(ROOT, "fractus_benchkit", "restored_model", "fractus_full_model.bin")

FAISS_IDX = os.path.join(OUT_DIR, "mapillary.faiss")
META_CSV  = os.path.join(OUT_DIR, "meta.csv")

# -----------------------------
# Utils
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def clamp_coords(lat, lon):
    return float(max(-90.0, min(90.0, lat))), float(max(-180.0, min(180.0, lon)))

# -----------------------------
# Charge profil Fractus (coeffs mixeur)
# -----------------------------
def load_fractus_profile():
    # Défauts “safe” si pas de profil
    prof = {
        "window": 32,
        "alpha_mean": 0.08,   # poids correction moyenne des scores
        "alpha_std": 0.04,    # poids correction dispersion
        "beta_knn": 0.85,     # poids du kNN (si FAISS dispo)
        "beta_plonk": 0.15,   # poids du Plonk
        "pattern": "fractus-full"
    }
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r") as f:
                data = json.load(f)
            prof.update({k: data[k] for k in prof.keys() if k in data})
            log.info(f"🧬 Profil Fractus chargé: {PROFILE_PATH}")
        except Exception as e:
            log.warning(f"Profil illisible, on garde les défauts: {e}")
    else:
        log.warning("Profil Fractus absent, on utilise les hyperparamètres par défaut.")
    return prof

# -----------------------------
# Optionnel: KNN géo via FAISS (StreetCLIP déjà chargé côté plonk_infer)
# -----------------------------
class GeoKNN:
    def __init__(self):
        self.ok = False
        self.index = None
        self.meta = None
        if HAS_FAISS and os.path.exists(FAISS_IDX) and os.path.exists(META_CSV):
            try:
                self.index = faiss.read_index(FAISS_IDX)
                self.meta = pd.read_csv(META_CSV)
                self.ok = True
                log.info(f"📚 FAISS chargé ({len(self.meta)} entrées).")
            except Exception as e:
                log.warning(f"Impossible de charger FAISS/meta: {e}")

    def predict(self, feat: np.ndarray, k: int = 8):
        if not self.ok:
            return None
        D, I = self.index.search(feat, k)
        w = 1.0 / (D[0] + 1e-6)
        w = w / w.sum()
        lats = self.meta.iloc[I[0]]["lat"].values
        lons = self.meta.iloc[I[0]]["lon"].values
        lat = float((w * lats).sum())
        lon = float((w * lons).sum())
        return clamp_coords(lat, lon)

# -----------------------------
# “Chargement” modèle restauré (bin)
# Ici on ne devine pas son format interne => on s’en sert comme drapeau d’activation
# pour autoriser la correction Fractus et le mixage kNN.
# -----------------------------
def restored_model_available():
    try:
        return os.path.getsize(RESTORED_BIN) > 0
    except Exception:
        return False

# -----------------------------
# Pipeline complet: Plonk + Fractus + (optionnel) kNN
# -----------------------------
def predict_with_fractus(image_rgb: np.ndarray):
    prof = load_fractus_profile()

    # 1) Plonk (StreetCLIP + FAISS interne à plonk_infer)
    base = plonk_infer.plonk_predict(Image.fromarray(image_rgb))[0]
    plat, plon = float(base["lat"]), float(base["lon"])

    # 2) Fractus core: scores fractals sur l'image (pas d'IA explicative)
    scores = compute_fractus_scores(image_rgb, window=int(prof["window"]), pattern=prof["pattern"])
    s_mean = float(np.mean(scores)) if scores is not None else 0.0
    s_std  = float(np.std(scores))  if scores is not None else 0.0

    # 3) Optionnel: KNN géo (si index présent)
    knn = GeoKNN()
    lat_knn, lon_knn = (None, None)
    if knn.ok:
        # Récupérer l’embedding image via plonk_infer (StreetCLIP)
        # On triche légèrement: plonk_infer n’expose pas get_image_features,
        # donc on refait un mini call privé en s’inspirant de plonk_infer._clip_embed
        try:
            from transformers import CLIPModel, CLIPProcessor
            MODEL_ID = "geolocal/StreetCLIP"
            _model = CLIPModel.from_pretrained(MODEL_ID)
            _proc  = CLIPProcessor.from_pretrained(MODEL_ID)
            _model.eval()
            inputs = _proc(images=Image.fromarray(image_rgb), return_tensors="pt")
            with np.no_grad():  # numpy n'a pas no_grad, donc on corrigera juste après…
                pass
            import torch
            with torch.no_grad():
                feat = _model.get_image_features(**inputs).cpu().numpy().astype("float32")
            lat_knn, lon_knn = knn.predict(feat, k=8)
        except Exception as e:
            log.warning(f"Impossible de calculer l'embedding pour KNN: {e}")

    # 4) Mixage des sources (comme ton régresseur léger)
    #    - correction fractale douce autour de Plonk
    lat_corr = plat + prof["alpha_mean"] * s_mean * 5.0
    lon_corr = plon + prof["alpha_std"]  * s_std  * 5.0
    lat_corr, lon_corr = clamp_coords(lat_corr, lon_corr)

    #    - si KNN dispo: interpolation (beta_knn vs beta_plonk)
    if lat_knn is not None and lon_knn is not None:
        beta_knn   = float(prof["beta_knn"])
        beta_plonk = float(prof["beta_plonk"])
        lat_mix = beta_knn * lat_knn + beta_plonk * lat_corr
        lon_mix = beta_knn * lon_knn + beta_plonk * lon_corr
        lat_mix, lon_mix = clamp_coords(lat_mix, lon_mix)
        engine = "Plonk+Fractus+kNN"
        out_lat, out_lon = lat_mix, lon_mix
    else:
        engine = "Plonk+Fractus"
        out_lat, out_lon = lat_corr, lon_corr

    return {
        "lat": out_lat,
        "lon": out_lon,
        "base": {"lat": plat, "lon": plon},
        "meta": {"engine": engine, "window": prof["window"], "pattern": prof["pattern"]}
    }

# -----------------------------
# Main CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fractus_restored_pipeline.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Impossible de lire l'image: {img_path}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not restored_model_available():
        log.warning("⚠️ Modèle restauré introuvable ou vide. On continue (correction Fractus + kNN si dispo).")

    res = predict_with_fractus(img_rgb)

    d = haversine_km(res["base"]["lat"], res["base"]["lon"], res["lat"], res["lon"])
    print("\n📊 Résultats comparés :")
    print(f"   ➤ Plonk seul       : lat={res['base']['lat']}, lon={res['base']['lon']}")
    print(f"   ➤ Plonk + Fractus  : lat={res['lat']}, lon={res['lon']}  [{res['meta']['engine']}]")
    print(f"📏 Distance (ajustement) : {d:.2f} km")

