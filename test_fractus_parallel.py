import numpy as np
from backend.fractus_parallel import fractus_transform_parallel

def main():
    print("➡️ Test fractus_parallel...")

    # Image factice 256x256 bruit blanc
    img = (np.random.rand(256, 256) * 255).astype(np.uint8)

    try:
        scores = fractus_transform_parallel(img, workers=8)
        if scores is not None and len(scores) > 0:
            print(f"✅ fractus_parallel → {len(scores)} scores")
            print("Aperçu:", scores[:10])
        else:
            print("⚠️ fractus_parallel → aucun score produit")
    except Exception as e:
        print("❌ Erreur:", e)

if __name__ == "__main__":
    main()

