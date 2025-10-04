from auto_selector import auto_predict

def main():
    print("=== Test Auto Selector (Règles contexte) ===")

    for img in ["photo_test.jpg", "tikal.jpg", "st_vincent.jpg"]:
        print(f"\n>>> Image : {img}")
        flags, preds = auto_predict(img, verbose=True)
        print("[RESULT] Flags activés ->", flags)
        print("[RESULT] Prédictions :")
        for r in preds:
            print(" ", r)

if __name__ == "__main__":
    main()

