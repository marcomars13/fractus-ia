#!/bin/bash
set -e

# 📂 Chemin du projet et de l'environnement virtuel
PROJECT_DIR="/Users/marco/Projets/fractus-ia"
VENV="$PROJECT_DIR/.venv"

echo "🚀 Activation de l'environnement virtuel..."
source "$VENV/bin/activate"

echo "⬆️ Mise à jour de pip..."
python3 -m pip install --upgrade pip

echo "📦 Installation de PyTorch CPU (Mac ARM)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "📦 Installation de xformers (version stable)..."
pip install xformers==0.0.25

echo "📦 Installation de Dinov2 depuis GitHub (sans dépendances)..."
pip install git+https://github.com/facebookresearch/dinov2.git --no-deps

echo "✅ Installation terminée !"
python3 -c "import torch; from dinov2.models import dinov2_vits14; print('Torch:', torch.__version__, ' | Dinov2 OK')"

