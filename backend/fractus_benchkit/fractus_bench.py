import argparse, os, time, pathlib, platform, psutil, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Import Fractus réel si disponible, sinon on basculera sur mock si --use-mock ---
try:
    from mock_fractus import process_chunk as fractus_process_chunk
except Exception:
    fractus_process_chunk = None

SIZE_UNITS = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}

def parse_size(s: str) -> int:
    s = s.strip().upper()
    for unit in ("GB","MB","KB","B"):
        if s.endswith(unit):
            return int(float(s[:-len(unit)].strip()) * SIZE_UNITS[unit])
    return int(float(s) * SIZE_UNITS["MB"])

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def sysinfo():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": getattr(psutil.cpu_freq(), "current", None),
        "mem_total_gb": psutil.virtual_memory().total / (1024**3),
    }

def write_random_file(path, size_bytes, chunk_bytes, seed):
    rng = np.random.default_rng(seed)
    remaining = size_bytes
    bs = chunk_bytes
    with open(path, "wb") as f:
        while remaining > 0:
            n = min(remaining, bs)
            buf = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            f.write(buf)
            remaining -= n

def mock_process(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr = arr ^ 0x5A
    arr = np.roll(arr, 7)
    arr = ((arr.astype(np.uint16) * 3 + 17) % 256).astype(np.uint8)
    return arr
    # même logique que mock_fractus mais inline pour éviter l'import si besoin
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr = arr ^ 0x5A
    arr = np.roll(arr, 7)
    arr = (arr * 3 + 17) % 256
    return arr

def real_process(buf):
    global fractus_process_chunk
    if fractus_process_chunk is None:
        raise RuntimeError("Aucune fonction Fractus réelle importée; utilisez --use-mock ou fournissez votre module.")
    return fractus_process_chunk(buf)

def bench_size(size_bytes, repeats, chunksize, io_only, compute_only, use_mock, seed_base, outdir):
    ensure_dir(outdir)
    tmpdir = os.path.join(outdir, "tmp")
    ensure_dir(tmpdir)

    t_totals, t_writes, t_computes, t_reads_est, mbps = [], [], [], [], []

    for r in range(repeats):
        seed = seed_base + r
        filename = os.path.join(tmpdir, f"blob_{size_bytes}_{r}.bin")

        # 1) Génération + écriture disque (I/O write)
        t0_write = time.perf_counter()
        write_random_file(filename, size_bytes, chunksize, seed)
        t1_write = time.perf_counter()
        t_write = t1_write - t0_write

        # 2) Lecture + compute (streaming)
        t0_total = time.perf_counter()
        t_compute = 0.0

        with open(filename, "rb") as f:
            if compute_only:
                data = f.read()
                c0 = time.perf_counter()
                if not io_only:
                    buf = memoryview(data)
                    if use_mock:
                        _ = mock_process(buf)
                    else:
                        _ = real_process(buf)
                t_compute += time.perf_counter() - c0
            else:
                while True:
                    chunk = f.read(chunksize)
                    if not chunk:
                        break
                    if not io_only:
                        c0 = time.perf_counter()
                        buf = memoryview(chunk)
                        if use_mock:
                            _ = mock_process(buf)
                        else:
                            _ = real_process(buf)
                        t_compute += time.perf_counter() - c0

        t1_total = time.perf_counter()
        t_total = t1_total - t0_total  # lecture + éventuel compute

        # Estimation I/O read = (total lecture+compute) - compute
        t_read = max(0.0, t_total - t_compute)

        t_totals.append(t_write + t_total)  # tout compris
        t_writes.append(t_write)
        t_computes.append(t_compute)
        t_reads_est.append(t_read)
        mbps.append((size_bytes / (1024**2)) / max(1e-9, (t_write + t_total)))

    return {
        "t_total_med": float(np.median(t_totals)),
        "t_write_med": float(np.median(t_writes)),
        "t_read_med": float(np.median(t_reads_est)),
        "t_compute_med": float(np.median(t_computes)),
        "mbps_med": float(np.median(mbps)),
        "runs": len(t_totals),
    }

def plot_scaling(df, outpng):
    plt.figure()
    plt.plot(df["size_mb"], df["t_total_med"], marker="o")
    plt.xlabel("Taille (MB)")
    plt.ylabel("Temps total médian (s)")
    plt.title("Fractus — Scalabilité (temps vs taille)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpng, dpi=144)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["100MB","1GB","10GB"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunksize", type=str, default="32MB")
    parser.add_argument("--io-only", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mock", action="store_true", default=True)
    parser.add_argument("--config", type=str, default="bench_config.yaml")
    args = parser.parse_args()

    # Charger config si présente
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        sizes = cfg.get("sizes", args.sizes)
        repeats = int(cfg.get("repeats", args.repeats))
        chunksize = cfg.get("chunksize", args.chunksize)
        seed = int(cfg.get("seed", args.seed))
        io_only = bool(cfg.get("io_only", args.io_only))
        compute_only = bool(cfg.get("compute_only", args.compute_only))
        use_mock = bool(cfg.get("use_mock", args.use_mock))
    else:
        sizes, repeats, chunksize = args.sizes, args.repeats, args.chunksize
        seed, io_only, compute_only, use_mock = args.seed, args.io_only, args.compute_only, args.use_mock

    sizes_bytes = [parse_size(s) for s in sizes]
    chunksize_bytes = parse_size(chunksize)

    outdir = "results"
    ensure_dir(outdir)

    meta = sysinfo()
    rows = []
    for s in sizes_bytes:
        r = bench_size(s, repeats, chunksize_bytes, io_only, compute_only, use_mock, seed, outdir)
        rows.append({
            "size_bytes": s,
            "size_mb": s/(1024**2),
            "t_total_med": r["t_total_med"],
            "t_write_med": r["t_write_med"],
            "t_read_med": r["t_read_med"],
            "t_compute_med": r["t_compute_med"],
            "mbps_med": r["mbps_med"],
            "repeats": repeats,
            **meta
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "fractus_bench.csv")
    df.to_csv(csv_path, index=False)
    png_path = os.path.join(outdir, "scaling_plot.png")
    plot_scaling(df, png_path)

    print(f"[OK] Résultats: {csv_path}")
    print(f"[OK] Graphique: {png_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
