from constraints import Flags, predict_hybrid

def main():
    print("=== Test constraints.py ===")

    # Cas 1 : tous les flags OFF (doit se comporter comme avant)
    flags_off = Flags(use_solar=False, use_dem=False, use_calib=False)
    res_off = predict_hybrid("photo_test.jpg", flags_off)
    print("Flags OFF ->", res_off)

    # Cas 2 : calibration ON (ne change rien pour l’instant mais doit tourner)
    flags_calib = Flags(use_solar=False, use_dem=False, use_calib=True)
    res_calib = predict_hybrid("photo_test.jpg", flags_calib)
    print("Flags CALIB ->", res_calib)

    # Cas 3 : solar + dem ON (stub retourne 0.0 donc pas d’effet, mais doit marcher)
    flags_full = Flags(use_solar=True, use_dem=True, use_calib=True)
    res_full = predict_hybrid("photo_test.jpg", flags_full)
    print("Flags SOLAR+DEM+CALIB ->", res_full)

if __name__ == "__main__":
    main()

