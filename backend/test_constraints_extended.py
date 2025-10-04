from constraints import Flags, predict_hybrid, predict_multi

def main():
    print("=== Test Constraints Extended ===")
    img = "photo_test.jpg"

    flags = Flags(use_solar=1, use_dem=1, use_calib=1, use_multi=1, use_skyline=1)

    print("\n-- Test prédiction hybride --")
    res = predict_hybrid(img, flags)
    for r in res:
        print(r)

    print("\n-- Test multi-photos --")
    res_multi = predict_multi([img, img], flags)
    for i, r in enumerate(res_multi, 1):
        print(f"#{i} -> {r}")

if __name__ == "__main__":
    main()

