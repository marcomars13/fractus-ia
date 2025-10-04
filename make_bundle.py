import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "Projets" / "fractus-ia"
DIST = ROOT / "dist"
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
BUNDLE = DIST / f"fractus_bundle_{STAMP}"

# Dossiers internes
SRC = BUNDLE / "src"
CONF = BUNDLE / "config"
SCRIPTS = BUNDLE / "scripts"
DOCKER = BUNDLE / "docker"
DATA = BUNDLE / "data"

print(f"➡️ Création du bundle dans {BUNDLE}")
for d in [SRC, CONF, SCRIPTS, DOCKER, SRC / "backend", DATA]:
    d.mkdir(parents=True, exist_ok=True)

# --- fichiers à embarquer ---
FILES = [
    "batch_compare_full_parallel.py",
    "run_tests_full.py",
    "compare_fractus_full.py",
    "compare_fractus_versions.py",
    "plonk_model.py",
    "fractus_model_full.py",
    "collector_mapillary_grid.py",
    "skyline_enhancer.py",
    "memory.py",
    "backend/fractus_parallel.py",
    "backend/constraints.py",
    "data/ground_truth.csv",
]

for f in FILES:
    src = ROOT / f
    if src.exists():
        if f.startswith("backend/"):
            dst = SRC / "backend" / Path(f).name
        elif f.startswith("data/"):
            dst = DATA / Path(f).name
        else:
            dst = SRC / Path(f).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"📦 Ajouté: {f}")
    else:
        print(f"⚠️ Manquant (ignoré): {f}")

# --- requirements.txt ---
req = """requests
pandas
numpy
matplotlib
reportlab
tqdm
"""
(SRC.parent / "requirements.txt").write_text(req)

# --- .env exemple ---
env = """# Exemple de configuration
MAPILLARY_TOKEN=MLY|xxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxxxxxxx
IMG_DIR=data/mapillary_world/thumbs
"""
(CONF / ".env.example").write_text(env)

# --- scripts ---
(SCRIPTS / "install.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo "✅ Installation terminée."
""")
os.chmod(SCRIPTS / "install.sh", 0o755)

(SCRIPTS / "run_benchmark_world.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
python3 src/batch_compare_full_parallel.py
""")
os.chmod(SCRIPTS / "run_benchmark_world.sh", 0o755)

(SCRIPTS / "run_collector_world.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
export $(grep -v '^#' config/.env | xargs -I{} echo {})
python3 src/collector_mapillary_grid.py
""")
os.chmod(SCRIPTS / "run_collector_world.sh", 0o755)

# --- README ---
readme = """# Fractus Bundle 🚀

Ce bundle contient :
- Le benchmark **Plonk vs Fractus (v1, 8c, Full)**
- Le collecteur **Mapillary Monde (10k images + miniatures)**
- Scripts d'installation et d'exécution
- Un environnement prêt à être déplacé sur une autre machine

---

## 1) Installation
```bash
bash scripts/install.sh
cp config/.env.example .env
# édite .env et mets ton MAPILLARY_TOKEN

