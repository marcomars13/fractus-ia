import os

# Dossier où sont les blobs
BLOB_DIR = "/Users/marco/Projets/fractus-ia/backend/fractus_benchkit/results/tmp"

# Fichier de sortie
OUTPUT = "/Users/marco/Projets/fractus-ia/backend/fractus_restored_model.bin"

# Récupère les fichiers blobs triés par taille puis par nom
blobs = sorted(
    [f for f in os.listdir(BLOB_DIR) if f.startswith("blob_")],
    key=lambda x: (os.path.getsize(os.path.join(BLOB_DIR, x)), x)
)

print(f"🧩 {len(blobs)} morceaux trouvés")
print("   " + "\n   ".join(blobs))

# Concaténation
with open(OUTPUT, "wb") as outfile:
    for fname in blobs:
        path = os.path.join(BLOB_DIR, fname)
        print(f"➕ Ajout de {fname} ({os.path.getsize(path)/1e6:.2f} MB)")
        with open(path, "rb") as infile:
            outfile.write(infile.read())

print(f"✅ Fichier reconstruit : {OUTPUT}")

