import os

# 📂 Dossier contenant les blobs
blob_dir = os.path.expanduser("~/Projets/fractus-ia/backend/fractus_benchkit/results/tmp")

# 📝 Fichier de sortie recomposé
out_file = os.path.expanduser("~/Projets/fractus-ia/backend/fractus_benchkit/restored_model/fractus_full_model.bin")

# Lister et trier les blobs par nom pour respecter l'ordre
blobs = sorted([f for f in os.listdir(blob_dir) if f.startswith("blob_")])

# Recomposer le modèle
with open(out_file, "wb") as wfd:
    for blob in blobs:
        print(f"Ajout de {blob} ...")
        blob_path = os.path.join(blob_dir, blob)
        with open(blob_path, "rb") as fd:
            while True:
                chunk = fd.read(50 * 1024 * 1024)  # lire par blocs de 50 Mo
                if not chunk:
                    break
                wfd.write(chunk)

print(f"✅ Modèle recomposé : {out_file}")

