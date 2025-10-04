from multi_infer import run_multi_infer
try:
    from constraints import Flags
except Exception:
    Flags = None

def main():
    print("=== Test Multi-Photos (empilement) ===")
    # Tu peux dupliquer la même image pour le test, ou mettre 2-3 fichiers réels
    images = ["photo_test.jpg", "photo_test.jpg"]

    flags = Flags(use_solar=False, use_dem=False, use_calib=True) if Flags else None

    res = run_multi_infer(
        images,
        top_k=3,         # on fusionne top-3 de chaque image
        radius_km=50.0,  # clusteriser ce qui est à < 50 km
        flags=flags
    )
    for i, r in enumerate(res, 1):
        print(f"#{i} -> lat={r['lat']}, lon={r['lon']} | votes={r['votes']} | "
              f"mean={r['score_mean']:.3f} | median={r['score_median']:.3f} | final={r['score_final']:.3f}")

if __name__ == "__main__":
    main()

