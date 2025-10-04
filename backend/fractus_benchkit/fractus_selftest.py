import numpy as np
import fractus_eval  # ton module Fractus

def test_fractus():
    # Exemple d'entrée simple : une séquence binaire courte
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

    # Appel de ta fonction principale
    try:
        out = fractus_eval.full_triangle(bits)
        print("[OK] full_triangle a produit une sortie :")
        print(out)
    except Exception as e:
        print("[ERREUR] full_triangle a planté :", e)

    # Test via le wrapper process_chunk
    try:
        chunk = bits.tobytes()
        out2 = fractus_eval.process_chunk(chunk)
        print("\n[OK] process_chunk a produit une sortie :")
        print(out2)
    except Exception as e:
        print("[ERREUR] process_chunk a planté :", e)

if __name__ == "__main__":
    test_fractus()

