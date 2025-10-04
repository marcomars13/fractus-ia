import numpy as np
from memory_active import MemoryActive

def main():
    print("=== Test Mémoire Active ===")
    mem = MemoryActive()

    # Ajoutons 2 entrées avec vérité terrain
    vec1 = np.random.rand(32).astype(np.float32)
    mem.add_entry(vec1, pred={"lat": 10.0, "lon": 20.0}, truth={"lat": 10.1, "lon": 20.1})

    vec2 = np.random.rand(32).astype(np.float32)
    mem.add_entry(vec2, pred={"lat": 50.0, "lon": 60.0}, truth={"lat": 49.9, "lon": 60.2})

    # Ajoutons 1 entrée sans vérité terrain
    vec3 = np.random.rand(32).astype(np.float32)
    mem.add_entry(vec3, pred={"lat": -5.0, "lon": -10.0})

    # Requête proche de vec1
    query = vec1 + np.random.normal(0, 0.01, size=32).astype(np.float32)
    result = mem.auto_adjust(query, top_k=3)

    print("Résultat ajusté :", result)

if __name__ == "__main__":
    main()

