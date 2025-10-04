import os
import random
import csv
import numpy as np
from skyline_extractor import extract_skyline_signature
from fractus_skyline import skyline_to_fractus
from run_plonk_restored import run_plonk_restored as run_plonk_api

def fractus_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Distance Hamming normalisée entre deux séquences Fractus Skyline."""
    if seq1.shape != seq2.shape:
        raise ValueError("❌ Séquences de tailles différentes !")
    return np.mean(seq1 != seq2)

def geo_distance(p1: dict, p2: dict) -> float:
    """Distance géographique euclidienne approximative."""
    return np.sqrt((p1["lat"] - p2["lat"])**2 + (p1["lon"] - p2["lon"])**2)

def safe_run_plonk(img_path: str):
    """Lance Plonk en mode robuste, avec log complet."""
    try:
        out = run_plonk_api(img_path)
        print(f"🔎 Raw Plonk output ({os.path.basename(img_path)}): {out}")

        # Si c'est une liste → prend le premier élément
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            out = out[0]

        if isinstance(out, dict) and "lat" in out and "lon" in out:
            # Vérifie si c'est (0,0) → considéré comme invalide
            if abs(out["lat"]) < 1e-6 and abs(out["lon"]) < 1e-6:
                print(f"⚠️ Plonk a renvoyé (0,0) → ignoré")
                return None
            return out

        print(f"⚠️ Format inattendu Plonk pour {img_path}: {out}")
        return None

    except Exception as e:
        print(f"❌ Erreur Plonk pour {img_path}: {e}")
        return None

# 📂 Dossier des images
img_dir = "/Users/marco/mapillary_france_adaptive/thumbs"
files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

if len(files) < 2:
    raise ValueError("❌ Pas assez d'images pour comparer.")

# 🔄 Nombre de paires
N_PAIRS = 20
results = []

for _ in range(N_PAIRS):
    img1, img2 = random.sample(files, 2)
    path1 = os.path.join(img_dir, img1)
    path2 = os.path.join(img_dir, img2)

    # === 1. Skyline-Fractus ===
    seq1 = skyline_to_fractus(extract_skyline_signature(path1))
    seq2 = skyline_to_fractus(extract_skyline_signature(path2))
    dist_fractus = fractus_distance(seq1, seq2)

    # === 2. Plonk ===
    plonk1 = safe_run_plonk(path1)
    plonk2 = safe_run_plonk(path2)

    if plonk1 is not None and plonk2 is not None:
        dist_plonk = geo_distance(plonk1, plonk2)
        combo_score = 0.7 * dist_fractus + 0.3 * np.log1p(dist_plonk)
        plonk_latlon = f"({plonk1['lat']:.4f},{plonk1['lon']:.4f}) vs ({plonk2['lat']:.4f},{plonk2['lon']:.4f})"
    else:
        dist_plonk = None
        combo_score = dist_fractus  # fallback
        plonk_latlon = "None"

    results.append((combo_score, dist_fractus, dist_plonk, img1, img2, plonk_latlon))

# Trie
results.sort(key=lambda x: x[0])

print("\n🔝 Top 5 paires les plus proches (combo Skyline+Plonk) :")
for score, df, dp, i1, i2, latlon in results[:5]:
    print(f"  {i1} vs {i2} → Combo={score:.3f} | Fractus={df:.3f} | Plonk={dp} | {latlon}")

print("\n🔻 Top 5 paires les plus éloignées (combo Skyline+Plonk) :")
for score, df, dp, i1, i2, latlon in results[-5:]:
    print(f"  {i1} vs {i2} → Combo={score:.3f} | Fractus={df:.3f} | Plonk={dp} | {latlon}")

# === Sauvegarde CSV ===
csv_path = "/Users/marco/Projets/fractus-ia/backend/plonk_fractus_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image1", "Image2", "ComboScore", "FractusDist", "PlonkDist", "PlonkLatLon"])
    for score, df, dp, i1, i2, latlon in results:
        writer.writerow([i1, i2, score, df, dp, latlon])

print(f"\n📂 Résultats sauvegardés dans {csv_path}")

