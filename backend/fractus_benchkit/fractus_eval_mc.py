import numpy as np

# ---------------------------------------------------
# Réductions XOR optimisées (monocœur et multicœur)
# ---------------------------------------------------

def _xor_shift_single(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Décale et combine arr[:n-shift] et arr[shift:] par XOR.
    Version monocœur.
    """
    n = arr.size
    left  = arr[: n - shift]
    right = arr[shift : n]
    return left ^ right

def _xor_shift_parallel(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Version multicœur : découpe le XOR en segments parallèles.
    Fallback monocœur si joblib indisponible ou taille trop petite.
    """
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        n = arr.size
        left  = arr[: n - shift]
        right = arr[shift : n]
        out = np.empty(left.shape, dtype=np.uint8)

        n_jobs = max(1, min(multiprocessing.cpu_count(), 8))
        L = left.size
        if L < 1_000_000 or n_jobs == 1:
            out[:] = left ^ right
            return out

        # Coupe en tranches quasi égales
        edges = np.linspace(0, L, n_jobs + 1, dtype=np.int64)

        def worker(a, b):
            out[a:b] = left[a:b] ^ right[a:b]

        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(int(edges[i]), int(edges[i+1]))
            for i in range(n_jobs)
        )
        return out
    except Exception:
        return _xor_shift_single(arr, shift)

# ---------------------------------------------------
# Réduction fractale rapide (O(n log n))
# ---------------------------------------------------

def _triangle_reduce_fast(arr_bits: np.ndarray) -> np.uint8:
    """
    Applique la réduction triangle mod 2 sur arr_bits (0/1) en O(n log n).
    Équivalent à répéter : arr = (arr[:-1] + arr[1:]) % 2, n-1 fois,
    mais regroupé par puissances de deux (accéléré).
    """
    a = (arr_bits & 1).astype(np.uint8, copy=False)
    n = a.size
    if n == 0:
        return np.uint8(0)
    k = n - 1

    while k > 0:
        s = 1 << (k.bit_length() - 1)
        a = _xor_shift_parallel(a, s)
        k -= s
    return np.uint8(a[0])

# ---------------------------------------------------
# API attendue par le benchmark
# ---------------------------------------------------

def process_chunk(chunk) -> np.uint8:
    """
    Uniformise l'appel pour le benchmark.
    - Convertit chunk en tableau binaire (0/1 via parité)
    - Applique la réduction rapide O(n log n) multicœur
    - Retourne le bit final
    """
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(chunk, dtype=np.uint8)
    elif isinstance(chunk, np.ndarray):
        arr = chunk.view(np.uint8)
    else:
        arr = np.frombuffer(memoryview(chunk), dtype=np.uint8)

    bits = arr & 1
    return _triangle_reduce_fast(bits)

