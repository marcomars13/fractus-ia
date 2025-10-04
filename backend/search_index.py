import faiss
import numpy as np
from fractus_ultimate import extract_vector

def main():
    # Charger l'index et la liste d'images
    index = faiss.read_index("backend/fractus_global.index")
    with open("backend/fractus_global_images.txt") as f:
        image_list = [line.strip() for line in f.readlines()]

    print(f"📦 Index chargé: {index.ntotal} vecteurs")

    # Image test → modifie le chemin si besoin
    test_img = "/Users/marco/Desktop/Unknown-11.jpeg"

    print(f"🔎 Encodage image test: {test_img}")
    vec = extract_vector(test_img)
    if vec is None:
        print("⚠️ Impossible d'extraire le vecteur")
        return

    # Recherche des 5 plus proches voisins
    D, I = index.search(vec, k=5)

    print("\n🏆 Résultats recherche (plus proches voisins) :")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if 0 <= idx < len(image_list):
            print(f"{rank}. {image_list[idx]}  (distance={dist:.2f})")
        else:
            print(f"{rank}. [index hors limite] (distance={dist:.2f})")

if __name__ == "__main__":
    main()

