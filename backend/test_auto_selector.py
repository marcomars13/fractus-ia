from auto_selector import auto_predict

def main():
    print("=== Test Auto Selector ===")
    img = "photo_test.jpg"  # change par une autre image si besoin

    flags, preds = auto_predict(img)

    print("\nFlags activés :")
    print(flags)

    print("\nPrédictions :")
    for r in preds:
        print(r)

if __name__ == "__main__":
    main()

