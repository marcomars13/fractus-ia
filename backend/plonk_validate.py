from PIL import Image
from plonk import PlonkPipeline

def main():
    # 📂 Image test
    img_path = "/Users/marco/Desktop/Unknown-11.jpeg"

    # Charger image
    pil_img = Image.open(img_path).convert("RGB")

    # Charger le modèle Plonk officiel
    print("🚀 Chargement PlonkPipeline (YFCC)")
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC", device="cpu")

    # Prédiction via l'API pipeline (préprocessing intégré)
    pred = pipe([pil_img], batch_size=1)

    if hasattr(pred, "tolist"):
        pred = pred.tolist()[0]

    lat, lon = pred[0], pred[1]
    print(f"📍 Plonk prediction → lat={lat:.5f}, lon={lon:.5f}")

if __name__ == "__main__":
    main()

