import argparse, os, time, pathlib, sys, importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mock_fractus import process_chunk as mock_process

SIZE_UNITS = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}

def parse_size(s: str) -> int:
    s = s.strip().upper()
    for unit in ("GB","MB","KB","B"):
        if s.endswith(unit):
            return int(float(s[:-len(unit)].strip()) * SIZE_UNITS[unit])
    return int(float(s) * SIZE_UNITS["MB"])

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def randbytes(size_bytes, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=size_bytes, dtype=np.uint8).tobytes()

def bench(size_bytes, repeats, chunksize, processor, label, seed_base):
    rows = []
    for r in range(repeats):
        data = randbytes(size_bytes, seed_base+r)
        t0 = time.perf_counter()
        offset = 0
        while offset < size_bytes:
            chunk = memoryview(data[offset:offset+chunksize])
            _ = processor(chunk)
            offset += chunksize
        t1 = time.perf_counter()
        rows.append(t1 - t0)
    return {
        "label": label,
        "size_bytes": size_bytes,
        "size_mb": size_bytes/(1024**2),
        "time_med": float(np.median(rows)),
        "mbps": (size_bytes/(1024**2))/np.median(rows)
    }

def plot_compare(df, outpng):
    plt.figure()
    for label in df["label"].unique():
        sub = df[df["label"]==label]
        plt.plot(sub["size_mb"], sub["time_med"], marker="o", label=label)
    plt.xlabel("Taille (MB)")
    plt.ylabel("Temps médian (s)")
    plt.title("Mock vs Fractus — Scalabilité")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=144)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, required=True, help="Nom du module Fractus réel (sans .py)")
    ap.add_argument("--sizes", nargs="+", default=["100MB","1GB"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--chunksize", type=str, default="32MB")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Import dynamique du module réel
    try:
        real_mod = importlib.import_module(args.module)
        real_process = getattr(real_mod, "process_chunk")
    except ModuleNotFoundError:
        print(f"[ERREUR] Module '{args.module}.py' introuvable dans ce dossier.")
        sys.exit(1)
    except AttributeError:
        print(f"[ERREUR] Le module '{args.module}.py' n'a pas de fonction process_chunk(chunk).")
        sys.exit(1)

    sizes_bytes = [parse_size(s) for s in args.sizes]
    chunksize_bytes = parse_size(args.chunksize)

    outdir = "results"
    ensure_dir(outdir)

    results = []
    for s in sizes_bytes:
        results.append(bench(s, args.repeats, chunksize_bytes, mock_process, "Mock", args.seed))
        results.append(bench(s, args.repeats, chunksize_bytes, real_process, "Fractus", args.seed))

    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "fractus_compare.csv")
    df.to_csv(csv_path, index=False)

    png_path = os.path.join(outdir, "fractus_compare.png")
    plot_compare(df, png_path)

    print(f"[OK] Résultats: {csv_path}")
    print(f"[OK] Graphique: {png_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
