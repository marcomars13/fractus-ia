import os
import torch
import numpy as np

CHUNK_DIR = "extracted_chunks"

def try_torch(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            return f"✅ torch.load OK (dict, {len(obj)} clés)"
        return f"✅ torch.load OK (type={type(obj)})"
    except Exception as e:
        return f"❌ torch.load FAIL → {e}"

def try_numpy(path):
    try:
        arr = np.load(path, allow_pickle=True)
        return f"✅ numpy.load OK (shape={arr.shape}, dtype={arr.dtype})"
    except Exception as e:
        return f"❌ numpy.load FAIL → {e}"

def try_magic(path):
    with open(path, "rb") as f:
        head = f.read(32)
    hex_head = " ".join([f"{b:02x}" for b in head])
    ascii_head = "".join([chr(b) if 32 <= b < 127 else "." for b in head])
    return f"🔍 Magic head HEX={hex_head} | ASCII={ascii_head}"

if __name__ == "__main__":
    if not os.path.exists(CHUNK_DIR):
        print(f"❌ Dossier {CHUNK_DIR} introuvable.")
        exit(1)

    files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".bin")])
    if not files:
        print("❌ Aucun chunk trouvé.")
        exit(1)

    print(f"📂 Analyse de {len(files)} chunks dans {CHUNK_DIR}\n")

    for fname in files:
        path = os.path.join(CHUNK_DIR, fname)
        print(f"=== {fname} ({os.path.getsize(path)/1024/1024:.2f} MB) ===")
        print(try_magic(path))
        print(try_torch(path))
        print(try_numpy(path))
        print("")

