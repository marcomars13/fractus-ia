from auto_selector import auto_predict

def main():
    print("=== Test Auto Selector (DEBUG) ===")
    img = "photo_test.jpg"  # remplace par une autre image pour varier

    flags, preds = auto_predict(img, verbose=True)

    print("\n[RESULT] Flags activés ->", flags)
    print("[RESULT] Prédictions :")
    for r in preds:
        print(" ", r)

if __name__ == "__main__":
    main()

