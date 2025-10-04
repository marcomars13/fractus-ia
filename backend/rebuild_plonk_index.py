import os
import joblib
import numpy as np
from sklearn.neighbors import KDTree

def main():
    plonk_file = "data/mapillary_world/plonk_world_index.joblib"
    output_file = "data/mapillary_world/plonk_world_index_fixed.joblib"

    print(f"📦 Chargement index Plonk brut: {plonk_file}")
    obj = joblib.load(plonk_file)

    if not isinstance(obj, dict):
        raise TypeError(f"❌ Format inattendu: {type(obj)}")

    # Vérifie les clés dispo
    print(f"🔑 Clés disponibles: {list(obj.keys())}")
    latlons = np.array(obj.get("latlons"))
    filenames = np.array(obj.get("filenames"))

    if latlons is None or filenames is None:
        raise ValueError("❌ Impossible de trouver 'latlons' ou 'filenames' dans l'index Plonk.")

    print(f"📂 latlons: {latlons.shape}, filenames: {len(filenames)}")

    # Si features X absents → on ne peut pas les inventer
    # Mais au moins on construit un KDTree sur latlons
    X = obj.get("X", None)
    if X is None:
        print("⚠️ Pas de features 'X' → on reconstruit KDTree uniquement sur latlons.")
        X = latlons  # fallback bidon (ça permet de garder la structure cohérente)

    # Construction KDTree
    print("🌲 Construction KDTree...")
    kdtree = KDTree(X, metric="euclidean")

    # Sauvegarde enrichie
    joblib.dump(
        {
            "X": X,
            "latlons": latlons,
            "filenames": filenames,
            "kdtree": kdtree,
        },
        output_file
    )
    print(f"✅ Nouvel index Plonk sauvegardé: {output_file}")

    # Sanity check
    dist, ind = kdtree.query(X[:1], k=1)
    print(f"🔎 Sanity check: dist={dist[0][0]} idx={ind[0][0]} → filename={filenames[ind[0][0]]}")

if __name__ == "__main__":
    main()
