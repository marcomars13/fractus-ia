from plonk.pipe import PlonkPipeline

# 📂 Chemin local vers ton repo HuggingFace téléchargé
LOCAL_REPO = "/Users/marco/Projets/fractus-ia/models/plonk_yfcc"

print("🚀 Chargement du pipeline Plonk (local)...")
pipe = PlonkPipeline(model_path=LOCAL_REPO, device="cpu")
print("✅ Pipeline chargé")

# 📂 Image de test
image_path = "/Users/marco/mapillary_france_adaptive/thumbs/1000091355214658.jpg"

# 🔮 Prédiction
out = pipe.predict(image_path)
print("📍 Résultat Plonk officiel :", out)

