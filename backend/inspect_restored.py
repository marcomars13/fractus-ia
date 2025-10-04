import torch
import os

MODEL_PATH = "fractus_benchkit/restored_model/fractus_full_model.bin"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Fichier introuvable: {MODEL_PATH}")

print(f"🔍 Inspection du modèle: {MODEL_PATH}")

try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    print("✅ torch.load() a réussi")
    if isinstance(checkpoint, dict):
        # Si c'est un state_dict complet
        if "state_dict" in checkpoint:
            keys = list(checkpoint["state_dict"].keys())
            print(f"📦 Contient une clé 'state_dict' avec {len(keys)} paramètres")
            print("🔑 Exemples:", keys[:10])
        else:
            keys = list(checkpoint.keys())
            print(f"📦 Dictionnaire de {len(keys)} entrées")
            print("🔑 Exemples:", keys[:10])
    else:
        print(f"⚠️ Objet non standard: {type(checkpoint)}")

except Exception as e:
    print("❌ torch.load() a échoué")
    print("Erreur:", e)

