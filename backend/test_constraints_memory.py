from constraints import Flags, predict_hybrid, MEMORY_DB
import numpy as np

def main():
    print("=== Test Constraints avec Mémoire Active ===")
    img = "test_img.jpg"

    # Ajouter une entrée mémoire avec vérité terrain
    if MEMORY_DB:
        vec = np.random.rand(32).astype(np.float32)
        MEMORY_DB.add_entry(vec, pred={"lat": 0.0, "lon": 0.0}, truth={"lat": 42.0, "lon": 7.0})
    
    flags = Flags(use_solar=0, use_dem=0, use_calib=0, use_multi=0, use_skyline=0, use_vector=0, use_memory=1)

    res = predict_hybrid(img, flags)
    for r in res:
        print(r)

if __name__ == "__main__":
    main()

