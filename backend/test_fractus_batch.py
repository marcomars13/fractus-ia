# backend/test_fractus_batch.py
import os
from multiprocessing import Pool, cpu_count
from backend.fractus_core import run_fractus

IMG_DIR = "test_images"

def process_image(img_path):
    try:
        result = run_fractus(img_path)
        # ✅ Conversion numpy.float64 -> float natif Python
        lat, lon, meta = result
        lat = float(lat)
        lon = float(lon)
        meta["neighbors"] = [[float(x), float(y)] for x, y in meta.get("neighbors", [])]
        meta["distances"] = [float(d) for d in meta.get("distances", [])]
        return (img_path, (lat, lon, meta))
    except Exception as e:
        return (img_path, f"❌ erreur: {e}")

def main():
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    images.sort()
    n = min(20, len(images))

    print(f"🚀 Batch test Fractus sur {n} images avec {cpu_count()} cœurs")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, images[:n])

    print(f"\n✅ {len(results)} images traitées")
    for i, (path, res) in enumerate(results, 1):
        print(f"{i:02d}. {res}")

if __name__ == "__main__":
    main()

