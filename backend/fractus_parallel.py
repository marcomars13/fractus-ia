#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fractus_parallel.py
-------------------
Transforme une image en vecteur de "scores fractals" avec option multi-résolution
et exécution parallèle sûre (macOS/Spawn-friendly).

- Entrée : chemin d'image OU numpy.ndarray (H,W[,C]) uint8/float
- Sortie : np.ndarray de shape (4097,) ou (12291,) si multi=True
- Parallélise par échelles (1.0, 0.5, 0.25) quand multi=True
- Évite tout test ambigu sur des ndarrays (pas de `if arr:`)
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple, Optional, List

import numpy as np

try:
    import cv2  # pour chargement/resize rapide
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from PIL import Image

__all__ = ["fractus_transform_parallel", "compute_fractus_scores"]


# ------------------------------ Utils sûrs ------------------------------

def _load_image_any(img_or_path) -> np.ndarray:
    """
    Charge une image depuis un chemin OU retourne l'array si c'est déjà un ndarray.
    Retourne un grayscale float32 normalisé [0,1].
    """
    if img_or_path is None:
        raise ValueError("Image is None")

    if isinstance(img_or_path, np.ndarray):
        arr = img_or_path
        if arr.size == 0:
            raise ValueError("Empty ndarray provided")
        # convert to gray
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if _HAS_CV2:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] == 3 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                # assume RGB
                gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(arr.dtype)
        elif arr.ndim == 2:
            gray = arr
        else:
            # mono-channel but unknown shape
            gray = np.squeeze(arr)
            if gray.ndim != 2:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
    else:
        # path
        path = str(img_or_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path not found: {path}")
        if _HAS_CV2:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                raise ValueError(f"Failed to read image with OpenCV: {path}")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            with Image.open(path) as im:
                im = im.convert("L")
                gray = np.array(im)

    gray = gray.astype(np.float32)
    # normalisation robuste
    mn, mx = np.nanmin(gray), np.nanmax(gray)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        # fallback : juste scale /255 si plausible
        if gray.dtype != np.uint8:
            # tente une normalisation soft
            denom = (np.std(gray) + 1e-6)
            gray = (gray - np.mean(gray)) / denom
            gray = (gray - gray.min()) / max(1e-6, (gray.max() - gray.min()))
        else:
            gray = gray / 255.0
    else:
        gray = (gray - mn) / max(1e-6, (mx - mn))
    return gray


def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    h, w = gray.shape
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    if _HAS_CV2:
        return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image
        return np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((nw, nh), resample=Image.BILINEAR)).astype(np.float32) / 255.0


# ------------------------------ Core features ------------------------------

def _fft_polar_bins(gray: np.ndarray, rbins: int = 64, tbins: int = 64) -> np.ndarray:
    """
    FFT -> magnitude -> passage en coordonnées polaires -> pooling moyen dans
    rbins x tbins. Retourne un vecteur de longueur rbins*tbins + 1 (DC/global).

    Longueur = 4096 + 1 = 4097 (comme observé dans tes tests).
    """
    # fenêtre Hann pour réduire les effets de bord
    h, w = gray.shape
    hann_y = np.hanning(h).reshape(-1, 1)
    hann_x = np.hanning(w).reshape(1, -1)
    window = hann_y * hann_x
    g = gray * window

    # FFT
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F).astype(np.float32)

    cy, cx = h // 2, w // 2
    y = (np.arange(h) - cy)[:, None]
    x = (np.arange(w) - cx)[None, :]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)  # [-pi, pi]

    r_max = r.max() if r.size > 0 else 1.0
    r_norm = np.clip(r / max(1e-6, r_max), 0.0, 1.0)
    t_norm = (theta + np.pi) / (2 * np.pi)  # [0,1]

    # indices de bins
    r_idx = np.minimum((r_norm * rbins).astype(np.int32), rbins - 1)
    t_idx = np.minimum((t_norm * tbins).astype(np.int32), tbins - 1)

    # pooling moyen par bin
    feat = np.zeros((rbins, tbins), dtype=np.float64)
    count = np.zeros((rbins, tbins), dtype=np.int64)
    # vectorisé
    for i in range(rbins):
        mask_r = (r_idx == i)
        if not np.any(mask_r):
            continue
        # pour chaque theta bin
        for j in range(tbins):
            m = mask_r & (t_idx == j)
            if np.any(m):
                vals = mag[m]
                feat[i, j] = vals.mean()
                count[i, j] = vals.size

    vec = feat.flatten()
    # normalise pour avoir des valeurs stables ~[0,1]
    vmax = float(np.max(vec)) if vec.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vec = vec / vmax

    # Ajoute un terme global (énergie moyenne) pour arriver à 4097
    global_mean = float(np.mean(mag) / (vmax + 1e-6))
    vec4097 = np.concatenate(([global_mean], vec)).astype(np.float32)
    return vec4097


def compute_fractus_scores(gray: np.ndarray, multi: bool = False) -> np.ndarray:
    """
    Calcule les features fractals 4097 (mono) ou 3*4097 (multi).
    """
    if gray is None or gray.size == 0:
        raise ValueError("Invalid grayscale image: empty")

    # Toujours s'assurer 2D
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {gray.shape}")

    if not multi:
        return _fft_polar_bins(gray, 64, 64)

    # multi-résolution : 1.0, 0.5, 0.25
    scales = [1.0, 0.5, 0.25]
    feats: List[np.ndarray] = []
    for s in scales:
        g = gray if abs(s - 1.0) < 1e-9 else _resize_gray(gray, s)
        feats.append(_fft_polar_bins(g, 64, 64))
    return np.concatenate(feats, axis=0).astype(np.float32)


# ------------------------------ Parallèle (spawn-safe) ------------------------------

def _compute_for_scale(args: Tuple[np.ndarray, float]) -> np.ndarray:
    gray, scale = args
    g = gray if abs(scale - 1.0) < 1e-9 else _resize_gray(gray, scale)
    return _fft_polar_bins(g, 64, 64)


def fractus_transform_parallel(
    img_or_path,
    workers: Optional[int] = None,
    multi: bool = False
) -> np.ndarray:
    """
    API externe :
      - img_or_path : chemin ou ndarray
      - workers     : nb de workers (par défaut: env FRACTUS_WORKERS ou nb de CPU)
      - multi       : True -> concat (1.0, 0.5, 0.25) => 12291 features

    Notes spawn/multiprocessing:
      - On parallélise par échelle (3 tâches au max), donc inutile d’allouer
        plus de 3 workers.
      - Fallback en mode séquentiel si ProcessPool indisponible.
    """
    gray = _load_image_any(img_or_path)
    # clamp taille minimale
    if min(gray.shape) < 8:
        gray = _resize_gray(gray, 8.0 / min(gray.shape))

    if not multi:
        # mode simple : calcule direct (évite overhead de spawn)
        return compute_fractus_scores(gray, multi=False)

    # multi-résolution -> au plus 3 tâches
    scales = [1.0, 0.5, 0.25]
    tasks = [(gray, s) for s in scales]
    max_w = max(1, min(len(tasks), _get_default_workers(workers)))

    try:
        # ProcessPoolExecutor est spawn-safe si utilisé depuis un module importé
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=max_w) as ex:
            futs = {ex.submit(_compute_for_scale, t): idx for idx, t in enumerate(tasks)}
            for f in as_completed(futs):
                idx = futs[f]
                results[idx] = f.result()
        # concat dans l’ordre des scales
        return np.concatenate(results, axis=0).astype(np.float32)
    except Exception:
        # fallback séquentiel
        feats = [ _compute_for_scale(t) for t in tasks ]
        return np.concatenate(feats, axis=0).astype(np.float32)


def _get_default_workers(user_workers: Optional[int]) -> int:
    if user_workers is not None and user_workers > 0:
        return int(user_workers)
    env = os.getenv("FRACTUS_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count())
    except Exception:
        return 1


# ------------------------------ Debug local ------------------------------

if __name__ == "__main__":
    # petit auto-test (à exécuter manuellement si besoin)
    rng = np.random.default_rng(0)
    demo = (rng.random((256, 256)) * 255).astype(np.uint8)
    v1 = fractus_transform_parallel(demo, workers=4, multi=False)
    v3 = fractus_transform_parallel(demo, workers=4, multi=True)
    print("normal:", v1.shape, float(v1.mean()))
    print("multi :", v3.shape, float(v3.mean()))

