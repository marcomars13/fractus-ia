#!/bin/zsh
echo "⚡ Activation de l'environnement virtuel..."
source /Users/marco/Projets/fractus-ia/backend/.venv/bin/activate

echo "🚀 Lancement du backend avec le Python du venv..."
exec /Users/marco/Projets/fractus-ia/backend/.venv/bin/python -m uvicorn api_infer:app --reload --port 8000

