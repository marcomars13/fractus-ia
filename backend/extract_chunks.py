#!/usr/bin/env python3
import os

FILE_PATH = "/Users/marco/Projets/fractus-ia/backend/fractus_restored_model.bin"
OUTPUT_DIR = "extracted_chunks"
CHUNK_AROUND = 10 * 1024 * 1024  # 10 MB avant/après

# Offsets trouvés (remplace-les avec ceux de scan_results.log)
OFFSETS = [
    123456789,  # exemple
    987654321   # exemple
]

def extract_blocks(path, offsets, out_dir, radius=CHUNK_AROUND):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    size = os.path.getsize(path)
    print(f"🔍 Fichier total : {size/(1024**3):.2f} GB")

    with open(path, "rb") as f:
        for i, off in enumerate(offsets):
            start = max(0, off - radius)
            end = min(size, off + radius)
            out_path = os.path.join(out_dir, f"chunk_{i}_off{off}.bin")

            f.seek(start)
            data = f.read(end - start)

            with open(out_path, "wb") as out:
                out.write(data)

            print(f"✅ Bloc {i} extrait : {out_path} "
                  f"(taille {(end - start)/(1024**2):.2f} MB, offset {off})")

if __name__ == "__main__":
    extract_blocks(FILE_PATH, OFFSETS, OUTPUT_DIR)

