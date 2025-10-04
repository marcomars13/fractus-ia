import os, random
import cv2

def check_dataset(path, sample_size=20):
    print(f"\n📂 Vérification du dataset : {path}")
    if not os.path.exists(path):
        print("❌ Chemin introuvable")
        return

    files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total = len(files)
    print(f"   → {total} fichiers trouvés")

    if total == 0:
        print("⚠️ Aucun fichier image détecté")
        return

    sample = random.sample(files, min(sample_size, total))
    ok, fail = 0, 0

    for f in sample:
        fp = os.path.join(path, f)
        img = cv2.imread(fp)
        if img is None:
            print(f"   ❌ {f} illisible")
            fail += 1
        else:
            print(f"   ✅ {f} OK ({img.shape[1]}x{img.shape[0]})")
            ok += 1

    print(f"\nRésumé : {ok} OK / {fail} illisibles sur {len(sample)} échantillons")

if __name__ == "__main__":
    # Vérifie les deux bases connues
    check_dataset("/Users/marco/mapillary_france_adaptive/thumbs")
    check_dataset("/Users/marco/Projets/fractus-ia/data/mapillary_all/thumbs_clean")

