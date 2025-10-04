from time import perf_counter
from fractus_parallel import fractus_transform_parallel, fractus_score_image, _encode_image_as_sequence_fallback

def main():
    print("=== Bench Fractus Parallel ===")

    # Séquence synthétique stable (longue) pour bench
    seq = _encode_image_as_sequence_fallback("st_vincent.jpg") * 4  # ~16k+ caractères
    window = 64
    step = 1

    # Mono-coeur "simulateur" : workers=1
    t0 = perf_counter()
    scores_1 = fractus_transform_parallel(seq, window=window, step=step, workers=1)
    t1 = perf_counter()

    # Multi-coeurs auto
    scores_auto = None
    t2 = perf_counter()
    scores_auto = fractus_transform_parallel(seq, window=window, step=step, workers=None)
    t3 = perf_counter()

    print(f"- Longueur séquence: {len(seq)} | fenêtres: {len(scores_1)}")
    print(f"- 1 worker   : {t1 - t0:.3f}s")
    print(f"- auto-core  : {t3 - t2:.3f}s  (env FRACTUS_WORKERS pour forcer)")

    # Exemple "image" -> scores
    print("\n=== Exemple sur image (fallback encode) ===")
    for img in ["st_vincent.jpg", "mumbai.jpg", "st_brieuc.jpg"]:
        s0 = perf_counter()
        sc = fractus_score_image(img, window=64, step=1, workers=None)
        s1 = perf_counter()
        print(f"{img}: {len(sc)} scores en {s1 - s0:.3f}s | aperçu: {sc[:5]}")

if __name__ == "__main__":
    main()

