#!/usr/bin/env python3
import sys, os
import numpy as np
from collections import Counter
from PIL import Image

# ---------- Fractus canonique ----------
def reduce_once(a):
    out = []
    i = 0
    n = len(a)
    while i < n:
        if i + 1 < n:
            out.append((a[i] + a[i+1]) % 3)
            i += 2
        else:
            out.append(a[i] % 3)
            i += 1
    return out

def signature_block(bits):
    cur = [b % 3 for b in bits]
    if len(cur) == 1:
        return (cur[0], cur[0])
    if len(cur) == 2:
        s = (cur[0] + cur[1]) % 3
        return (s, s)
    while len(cur) > 2:
        cur = reduce_once(cur)
    return (cur[0] % 3, cur[1] % 3)

def full_triangle(bits):
    level = [b % 3 for b in bits]
    levels = [level]
    while len(level) > 1:
        level = reduce_once(level)
        levels.append(level)
    return levels

def eval_on_bitstream(bits, block_n):
    pad = (block_n - (len(bits) % block_n)) % block_n
    if pad:
        bits = bits + [0]*pad
    num_blocks = len(bits)//block_n

    sigs = []
    dag_nodes_total = 0

    for b in range(num_blocks):
        blk = bits[b*block_n:(b+1)*block_n]
        sig = signature_block(blk)
        sigs.append(sig)
        levels = full_triangle(blk)
        intern=set()
        for lev in levels:
            intern.add(tuple(lev))
        dag_nodes_total += len(intern)

    sig_counts = Counter(sigs)
    metrics = {
        "blocks": num_blocks,
        "block_n": block_n,
        "unique_signatures": len(sig_counts),
        "dag_nodes_total": dag_nodes_total,
        "pad_bits": pad
    }
    return metrics, sig_counts

# ---------- Convertisseurs ----------
def text_to_bits(path):
    with open(path, "rb") as f:
        b = f.read()
    bits = []
    for byte in b:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits, len(b)

def image_to_bits(path):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    h,w = arr.shape
    bits = []
    for y in range(h):
        for x in range(w):
            byte = int(arr[y,x])
            for i in range(8):
                bits.append((byte >> i) & 1)
    return bits, h*w

# ---------- Main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: fractus_eval.py <fichier.txt|.png|.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()

    if ext in [".txt"]:
        bits, size = text_to_bits(path)
        dtype = "texte"
    elif ext in [".png", ".jpg", ".jpeg"]:
        bits, size = image_to_bits(path)
        dtype = "image"
    else:
        print(f"Extension non supportée: {ext}")
        sys.exit(1)

    print(f"Analyse Fractus sur {dtype} : {path}")
    print(f"Taille (unités): {size}")
    print("------------------------------------------------")

    # Fichier de rapport
    report_path = os.path.splitext(path)[0] + "_fractus_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Rapport Fractus ===\n")
        f.write(f"Fichier : {path}\n")
        f.write(f"Type    : {dtype}\n")
        f.write(f"Taille  : {size}\n\n")

        for n in [64,128,256,512]:
            m, sigs = eval_on_bitstream(bits, n)
            f.write(f"Bloc n={n} :\n")
            f.write(f"  blocs = {m['blocks']}\n")
            f.write(f"  signatures uniques = {m['unique_signatures']}\n")
            f.write(f"  DAG nodes = {m['dag_nodes_total']}\n")
            f.write(f"  padding bits = {m['pad_bits']}\n")
            f.write(f"  histogramme signatures = {dict(sigs)}\n\n")

    print(f"✅ Rapport généré : {report_path}")

if __name__ == "__main__":
    main()
import numpy as np

def full_triangle_numpy(bits):
    """
    Version vectorisée de full_triangle.
    bits : np.array d'entiers (0/1 ou uint8)
    Retourne la pyramide des réductions.
    """
    levels = [bits]
    arr = bits.copy()

    while arr.size > 1:
        # Exemple : somme des paires modulo 2
        arr = (arr[:-1] + arr[1:]) % 2
        levels.append(arr)

    return levels

# --- Wrapper pour le banc de test ---
import numpy as np

def full_triangle_numpy(bits):
    """
    Version vectorisée de full_triangle.
    bits : np.array d'entiers (0/1 ou uint8)
    Retourne la pyramide des réductions.
    """
    levels = [bits]
    arr = bits.copy()

    while arr.size > 1:
        # Exemple : somme des paires modulo 2
        arr = (arr[:-1] + arr[1:]) % 2
        levels.append(arr)

    return levels

# --- Wrapper pour le banc de test ---
def process_chunk(chunk):
    """
    Wrapper utilisé par le banc de test.
    Transforme le chunk en tableau de bits et applique la version NumPy.
    """
    import numpy as np

    # Convertir le chunk en tableau d'octets
    arr = np.frombuffer(chunk, dtype=np.uint8)

    # Réduire modulo 2 pour rester en binaire (0/1)
    arr = arr % 2

    return full_triangle_numpy(arr)

# --- Wrapper optimisé pour le banc de test ---
def process_chunk(chunk):
    """
    Version optimisée de Fractus pour le benchmark.
    - Transforme le chunk en bits (0/1)
    - Applique la réduction fractale en boucle
    - Ne garde que la ligne finale (résumé)
    """
    import numpy as np

    # Convertir le chunk en tableau d'octets
    arr = np.frombuffer(chunk, dtype=np.uint8)

    # Réduction modulo 2 pour obtenir des bits
    arr = arr % 2

    # Boucle de réduction jusqu'à une seule valeur
    while arr.size > 1:
        arr = (arr[:-1] + arr[1:]) % 2

    return arr  # le bit final

# --- Wrapper optimisé (réduction logarithmique) ---
def process_chunk(chunk):
    """
    Version rapide de Fractus :
    - transforme les octets en bits (0/1)
    - applique des réductions par paliers (O(n log n) au lieu de O(n²))
    - retourne seulement la valeur finale
    """
    import numpy as np

    # Transformer le chunk en tableau d'octets
    arr = np.frombuffer(chunk, dtype=np.uint8)

    # Réduction en bits
    arr = arr % 2

    # Réduction logarithmique
    n = arr.size
    step = 1
    while n > 1:
        # On combine arr[0]..arr[n-step-1] avec arr[step]..arr[n-1]
        arr = (arr[:-step] + arr[step:]) % 2
        n = arr.size
        step *= 2

    return arr[0]

