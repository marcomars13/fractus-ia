from vector_match import dummy_encode, VECTOR_DB

def main():
    # Encodage factice
    vec = dummy_encode("test.jpg")
    print("Vecteur encodé :", vec)

    # Recherche dans la base
    results = VECTOR_DB.search(vec, top_k=3)
    print("Résultats de recherche :")
    for r in results:
        print(f" - {r['label']} ({r['lat']}, {r['lon']}) -> dist={r['distance']:.4f}")

if __name__ == "__main__":
    main()

