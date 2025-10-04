#!/usr/bin/env python3
import os

FILE_PATH = "/Users/marco/Projets/fractus-ia/backend/fractus_restored_model.bin"
CHUNK_SIZE = 1024 * 1024  # 1 MB
LOG_FILE = "scan_results.log"

MAGIC_SIGS = {
    b"\x80\x04": "Pickle (PyTorch/Joblib)",
    b"PK\x03\x04": "ZIP/PK",
    b"\x93NUMPY": "NumPy .npy",
    b"FAISS": "FAISS index",
    b"PyTorch": "PyTorch checkpoint string",
    b"torch": "Torch reference",
    b"PT": "Possible PyTorch weights",
    b"HDF": "HDF5",
    b"\x1f\x8b": "Gzip"
}

def scan_file(path, chunk_size=CHUNK_SIZE, log_file=LOG_FILE):
    size = os.path.getsize(path)
    print(f"🔍 Scan du fichier {path}")
    print(f"📏 Taille totale : {size/(1024**3):.2f} GB")

    found = []
    with open(path, "rb") as f, open(log_file, "w") as log:
        offset = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for sig, desc in MAGIC_SIGS.items():
                idx = chunk.find(sig)
                if idx != -1:
                    absolute = offset + idx
                    line = f"✨ {desc} @ offset {absolute:,} (≈ {absolute/(1024**2):.2f} MB)\n"
                    print(line.strip())
                    log.write(line)
                    found.append((absolute, desc))

            offset += len(chunk)

    print(f"\n✅ Résultats enregistrés dans {log_file}")
    if not found:
        print("❌ Aucune signature connue trouvée.")
    else:
        print(f"⚡ {len(found)} signatures détectées.")

if __name__ == "__main__":
    scan_file(FILE_PATH)

