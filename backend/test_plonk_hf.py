from plonk.pipe import PlonkPipeline
from PIL import Image
import os

# 📂 chemin vers ton modèle téléchargé
MODEL_PATH = "./models/plonk_yfcc/pytorch_model.bin"

# ⚡ Charger le pipeline avec les poids locaux
pipeline = PlonkPipeline(
    model_path=MODEL_PATH,
    device="cpu"
)

# 🖼️ Test sur une image Mapillary
img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
test_img = os.path.join(img_dir, os.listdir(img_dir)[0])
img = Image.open(test_img).convert("RGB")

out = pipeline(img)

print("✅ Résultat Plonk officiel :", out)

