from skyline_enhancer import skyline_match

def main():
    print("=== Test Skyline Enhancer ===")
    image = "photo_test.jpg"  # tu peux changer pour tester une autre photo
    try:
        res = skyline_match(image)
        print(f"Image: {image}")
        print(f"→ Meilleur label : {res['label_best']} (distance {res['distance']:.4f})")
        print(f"Profil extrait : {res['profile_len']} colonnes")
    except Exception as e:
        print("Erreur :", e)

if __name__ == "__main__":
    main()

