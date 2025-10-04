import os
from plonk_model import run_plonk_api

# 📂 Dossier test
img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

if not files:
    raise ValueError("❌ Aucun fichier trouvé dans le dossier !")

# Prend la première image
test_img = os.path.join(img_dir, files[0])

print(f"🖼️ Test Plonk sur {test_img}")

try:
    out = run_plonk_api(test_img)
    print("✅ Retour brut Plonk :", out)

    # Vérification plus détaillée
    if isinstance(out, list):
        print(f"ℹ️ Plonk a renvoyé une liste de {len(out)} éléments.")
        if len(out) > 0:
            print("🔎 Premier élément :", out[0])

    elif isinstance(out, dict):
        print("ℹ️ Plonk a renvoyé un dict avec clés :", list(out.keys()))
        if "lat" in out and "lon" in out:
            print(f"🌍 Coordonnées : lat={out['lat']}, lon={out['lon']}")

    else:
        print("⚠️ Format inattendu :", type(out))

except Exception as e:
    print(f"❌ Erreur lors de l’appel à Plonk : {e}")

