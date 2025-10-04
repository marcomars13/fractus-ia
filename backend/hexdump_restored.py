#!/usr/bin/env python3
import os

FILE_PATH = "fractus_benchkit/restored_model/fractus_full_model.bin"
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB

def hexdump(data, length=16):
    """Retourne une chaîne hex + ascii lisible."""
    result = []
    for i in range(0, len(data), length):
        chunk = data[i:i+length]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        ascii_bytes = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        result.append(f"{i:08x}  {hex_bytes:<{length*3}}  {ascii_bytes}")
    return "\n".join(result)

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Fichier introuvable: {FILE_PATH}")
        return

    size = os.path.getsize(FILE_PATH)
    print(f"🔍 Fichier: {FILE_PATH}")
    print(f"📏 Taille totale: {size/1024/1024:.2f} MB\n")

    with open(FILE_PATH, "rb") as f:
        # Lire début
        head = f.read(CHUNK_SIZE)
        print("=== Début du fichier ===")
        print(hexdump(head[:512]))  # juste 512 premiers octets
        print("...")

        # Lire fin
        f.seek(max(0, size - CHUNK_SIZE))
        tail = f.read(CHUNK_SIZE)
        print("\n=== Fin du fichier ===")
        print(hexdump(tail[-512:]))  # derniers 512 octets
        print("...")

if __name__ == "__main__":
    main()

