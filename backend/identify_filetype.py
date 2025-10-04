#!/usr/bin/env python3
import os

FILE_PATH = "/Users/marco/Projets/fractus-ia/backend/fractus_restored_model.bin"

print(f"🔍 Analyse du fichier : {FILE_PATH}")

if not os.path.exists(FILE_PATH):
    print("❌ Fichier introuvable.")
    exit(1)

size = os.path.getsize(FILE_PATH)
print(f"📏 Taille totale : {size / (1024**3):.2f} GB")

with open(FILE_PATH, "rb") as f:
    header = f.read(64)  # On lit les 64 premiers octets pour inspection

# Affichage hexadécimal
hex_bytes = " ".join(f"{b:02x}" for b in header)
ascii_bytes = "".join(chr(b) if 32 <= b < 127 else "." for b in header)

print("\n=== En-tête brut (64 octets) ===")
print(f"HEX   : {hex_bytes}")
print(f"ASCII : {ascii_bytes}")
print("===============================")

