from plonk import PlonkPipeline
from PIL import Image
from fractus_ultimate import run_fractus_ultimate
import math

# 📂 Chemin de l'image Cappadoce
img_path = "/Users/marco/Desktop/Unknown-11.jpeg"

# Coordonnées approximatives Cappadoce (Göreme)
gt_lat, gt_lon = 38.65, 34.83

# Fonction haversine (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Charger le modèle Plonk
pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC", device="cpu")

# Prétraitement gagnant → resize_384
pil_img = Image.open(img_path).convert("RGB")
pil_img = pil_img.resize((384,384))

# 🔮 Prédiction Plonk
plonk_pred = pipe([pil_img])
lat, lon = plonk_pred[0].tolist()
err_plonk = haversine(gt_lat, gt_lon, lat, lon)
print(f"📍 Plonk: lat={lat:.5f}, lon={lon:.5f} | erreur={err_plonk:.2f} km")

# 🔮 Prédiction Fractus
fractus_pred = run_fractus_ultimate(img_path)
lat_f, lon_f = fractus_pred["lat"], fractus_pred["lon"]
err_fractus = haversine(gt_lat, gt_lon, lat_f, lon_f)
print(f"📍 Fractus: lat={lat_f:.5f}, lon={lon_f:.5f} | erreur={err_fractus:.2f} km")

