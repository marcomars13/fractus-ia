# Mock Fractus: pipeline vectoriel simple pour simuler un traitement chunk-wise
import numpy as np

def process_chunk(chunk):
    """
    chunk: bytes | memoryview | np.ndarray
    Retourne une copie transformée (simulation de calcul streaming).
    """
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(chunk, dtype=np.uint8)
    elif isinstance(chunk, np.ndarray):
        arr = chunk.view(np.uint8)
    else:
        raise TypeError("Type de chunk non supporté")

    # Opérations simples et rapides pour simuler un calcul
    arr = arr ^ 0x5A
    arr = np.roll(arr, 7)
    # ⚠️ correction : on passe temporairement en uint16 pour éviter l'overflow
    arr = ((arr.astype(np.uint16) * 3 + 17) % 256).astype(np.uint8)
    return arr.tobytes()

