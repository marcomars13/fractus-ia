import os
from fractus_core import fractus_transform
from fractus_parallel import _encode_image_as_sequence_fallback

def main():
    print("=== Test Fractus Core Multi-Resolution ===")
    seq = _encode_image_as_sequence_fallback("st_vincent.jpg")

    # Normal single-window
    os.environ["FRACTUS_WORKERS"] = "1"
    os.environ["FRACTUS_MULTI"] = "0"
    sc1 = fractus_transform(seq, window=64, step=1)
    print(f"[Normal] {len(sc1)} scores | aperçu: {sc1[:5]}")

    # Multi-resolution
    os.environ["FRACTUS_MULTI"] = "1"
    sc2 = fractus_transform(seq, window=64, step=1)
    print(f"[Multi-res] {len(sc2)} scores | aperçu: {sc2[:5]}")

if __name__ == "__main__":
    main()

