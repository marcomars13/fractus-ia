#!/usr/bin/env python3
import re
from pathlib import Path

# 🔎 Fichiers candidats à scanner (existe → on scanne, sinon on ignore)
FILES = [
    Path("backend/fractus_model_full.py"),
    Path("fractus_model_full.py"),
    Path("backend/fractus_restored_pipeline.py"),
    Path("backend/fractus_parallel.py"),
    Path("backend/fractus_benchkit/fractus_eval.py"),
    Path("backend/skyline_enhancer.py"),
    Path("backend/memory.py"),
]

KEYS = ["enhance_skyline", "vector_match", "multi", "Memory"]

def scan_file(p: Path):
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"⚠️  Impossible de lire {p}: {e}")
        return

    lines = src.splitlines()

    defs = []
    classes = []
    imports = []
    matches = {k: [] for k in KEYS}

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith("def "):
            defs.append(f"{i}: {s}")
        elif s.startswith("class "):
            classes.append(f"{i}: {s}")
        elif s.startswith("import ") or s.startswith("from "):
            imports.append(f"{i}: {s}")
        for k in KEYS:
            if k.lower() in s.lower():
                matches[k].append(f"{i}: {s}")

    print(f"\n📄 {p}  ({p.stat().st_size} octets)")
    if classes:
        print("  🏷️ Classes :")
        for c in classes:
            print("   ", c)
    if defs:
        print("  ⚙️ Fonctions :")
        for d in defs:
            print("   ", d)
    if imports:
        print("  📦 Imports :")
        for imp in imports:
            print("   ", imp)

    # Résumé mots-clés
    print("  🔑 Mots-clés :")
    for k in KEYS:
        if matches[k]:
            print(f"   ✅ {k} présent ({len(matches[k])} occurrence(s))")
            # Montre jusqu'à 3 lignes trouvées
            for m in matches[k][:3]:
                print("     •", m)
            if len(matches[k]) > 3:
                print(f"     … +{len(matches[k]) - 3} autres")
        else:
            print(f"   ❌ {k} absent")

if __name__ == "__main__":
    found_any = False
    for f in FILES:
        if f.exists():
            found_any = True
            scan_file(f)
    if not found_any:
        print("❌ Aucun des fichiers cibles n'existe dans ce projet. Vérifie le chemin.")


