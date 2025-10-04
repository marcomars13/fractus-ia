import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plonk_model import run_plonk_api

# prends une image du dossier Monde
img_path = "data/mapillary_world/thumbs_clean/1001030207387286.jpg"

res = run_plonk_api(img_path, return_features=True)

print("✅ Résultat test Plonk avec return_features=True :")
print(res)


