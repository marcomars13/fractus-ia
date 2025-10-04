import cv2
from PIL import Image
from fractus_benchkit.restore_utils import load_restored_model  # charge le modèle recomposé
import plonk_infer

# Charger le modèle restauré
fractus_model = load_restored_model("/Users/marco/Projets/fractus-ia/backend/fractus_benchkit/restored_model/fractus_full_model.bin")

# Chemin de l'image test
img_path = "/Users/marco/Desktop/Unknown-2.jpeg"

# Lire l'image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 🔹 Plonk seul
plonk_pred = plonk_infer.plonk_predict(Image.fromarray(img_rgb))
print("📍 Plonk seul →", plonk_pred)

# 🔹 Plonk + Fractus
fractus_pred = fractus_model.predict(img_rgb)  # ou la méthode exacte selon ton ancien code
print("📍 Plonk + Fractus →", fractus_pred)

