import os
import random
import csv
from datetime import datetime
from fractus_skyline import run_fractus_skyline

IMG_DIR = "/Users/marco/mapillary_france_adaptive/thumbs"
CSV_PATH = "backend/fractus_skyline_scores.csv"

def main():
    # Récupérer 10 images aléatoires
    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if len(images) == 0:
        raise FileNotFoundError(f"❌ Aucune image trouvée dans {IMG_DIR}")

    sample_images = random.sample(images, min(10, len(images)))

    results = []
    for img_name in sample_images:
        img_path = os.path.join(IMG_DIR, img_name)
        print(f"\n🖼️ Test image : {img_name}")

        out = run_fractus_skyline(img_path)
        lat, lon, score = out.get("lat"), out.get("lon"), out.get("score", 0.0)

        print(f"   🔮 Fractus Skyline → lat={lat}, lon={lon}, score={score:.6f}")

        results.append([
            datetime.now().isoformat(),
            img_name,
            lat,
            lon,
            score
        ])

    # Sauvegarde CSV
    header = ["timestamp", "image", "lat", "lon", "score"]
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(results)

    print(f"\n📂 Résultats sauvegardés dans {CSV_PATH}")

if __name__ == "__main__":
    main()

