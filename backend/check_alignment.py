import os
import joblib

def main():
    fractus_file = "data/mapillary_world/ground_truth_world_for_partial_index.joblib"
    filenames_file = "data/mapillary_world/fractus_filenames.txt"

    print(f"📦 Chargement index Fractus: {fractus_file}")
    obj = joblib.load(fractus_file)
    if isinstance(obj, tuple):
        features, coords = obj
    elif isinstance(obj, dict):
        features, coords = obj["features"], obj["coords"]
    else:
        raise TypeError(f"❌ Format inattendu: {type(obj)}")

    print(f"🔢 Features: {features.shape}, Coords: {coords.shape}")

    if not os.path.exists(filenames_file):
        raise FileNotFoundError(f"❌ Fichier manquant: {filenames_file}")

    with open(filenames_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"📝 {len(filenames)} filenames chargés")

    if features.shape[0] != len(filenames):
        print("⚠️ Incohérence !")
        print(f"   Features: {features.shape[0]} vs Filenames: {len(filenames)}")
        # Affiche 5 premiers pour inspection
        print("Exemples features idx → filename:")
        for i in range(min(5, len(filenames))):
            print(f"   {i}: {filenames[i]}")
    else:
        print("✅ Alignement parfait: features et filenames ont la même taille.")

if __name__ == "__main__":
    main()

