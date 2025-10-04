import os
import torch
import traceback

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fractus_restored_model.bin")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Fichier introuvable: {MODEL_PATH}")
    exit(1)

print(f"🧪 Tentative de chargement TorchScript: {MODEL_PATH}")

try:
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    print("✅ Succès: TorchScript model chargé.")
    print("📋 Type:", type(model))
    print("🔎 Attributs disponibles:", dir(model)[:20])  # affiche les 20 premiers attributs
except Exception as e:
    print("❌ Échec du chargement TorchScript")
    print("Erreur:", str(e))
    traceback.print_exc()

