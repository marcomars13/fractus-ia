import os
from time import perf_counter
from fractus_core import fractus_transform
from fractus_parallel import _encode_image_as_sequence_fallback

def main():
    print("=== Test Wrapper Fractus Core ===")
    seq = _encode_image_as_sequence_fallback("st_vincent.jpg") * 2

    # Mono-cœur
    os.environ["FRACTUS_WORKERS"] = "1"
    t0 = perf_counter()
    s1 = fractus_transform(seq, window=64, step=1)
    t1 = perf_counter()
    print(f"[Mono] {len(s1)} scores en {t1-t0:.3f}s | aperçu: {s1[:5]}")

    # Multi-cœur (forcé à 8)
    os.environ["FRACTUS_WORKERS"] = "8"
    t2 = perf_counter()
    s2 = fractus_transform(seq, window=64, step=1)
    t3 = perf_counter()
    print(f"[Multi] {len(s2)} scores en {t3-t2:.3f}s | aperçu: {s2[:5]}")

if __name__ == "__main__":
    main()

