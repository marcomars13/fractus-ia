from spectral_analyzer import analyze_spectrum

def main():
    print("=== Test Analyse Spectrale FFT ===")
    images = ["photo_test.jpg", "tikal.jpg", "angkor.jpg"]

    for img in images:
        res = analyze_spectrum(img)
        print(f"\nImage : {res['image']}")
        print(f"- Longueur signature : {res['signature_len']}")
        print(f"- Fréquence dominante : {res['dom_freq']} (index {res['dom_index']})")
        print(f"- Catégorie heuristique : {res['category']}")

if __name__ == "__main__":
    main()

