from constraints import Flags, predict_hybrid

def main():
    print("=== Test Constraints avec Analyse Spectrale ===")
    img = "angkor.jpg"

    flags = Flags(use_solar=0, use_dem=0, use_calib=0, use_multi=0,
                  use_skyline=0, use_vector=0, use_memory=0, use_spectral=1)

    res = predict_hybrid(img, flags)
    for r in res:
        print(r)

if __name__ == "__main__":
    main()

