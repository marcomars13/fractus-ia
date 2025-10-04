from fallback_fractus import fallback_predict

def main():
    print("=== Test Fallback Fractus ===")

    img = "photo_test.jpg"

    # Test sans contraintes
    res1 = fallback_predict(img, {"solar": False, "dem": False, "biome": False})
    print("Sans contraintes :", res1)

    # Test avec contraintes physiques
    res2 = fallback_predict(img, {"solar": True, "dem": True, "biome": True})
    print("Avec contraintes :", res2)

if __name__ == "__main__":
    main()

