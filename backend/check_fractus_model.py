import torch
import os

MODEL_PATH = "/Users/marco/Projets/fractus-ia/backend/fractus_restored_model.bin"

print(f"🔍 Vérification du modèle : {MODEL_PATH}")
print(f"📏 Taille : {os.path.getsize(MODEL_PATH) / (1024**3):.2f} GB")

try:
    obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    print("✅ torch.load() a réussi")
    if isinstance(obj, dict):
        print(f"Clés principales : {list(obj.keys())[:10]}")
    else:
        print(f"Type d’objet : {type(obj)}")
except Exception as e:
    print("❌ torch.load() a échoué")
    print("Erreur :", e)

