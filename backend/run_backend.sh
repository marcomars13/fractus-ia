#!/bin/bash
# 🚀 Script de lancement backend Fractus+Plonk

PROJECT_DIR="$HOME/Projets/fractus-ia"
VENV_DIR="$PROJECT_DIR/.venv"
BACKEND_DIR="$PROJECT_DIR/backend"
PORT=8000

echo "🛑 Vérification des processus sur le port $PORT..."
PIDS=$(lsof -ti :$PORT)
if [ ! -z "$PIDS" ]; then
  echo "➡️  Kill des process: $PIDS"
  kill -9 $PIDS
else
  echo "✅ Aucun process bloquant sur le port $PORT"
fi

echo "⚡ Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

echo "🚀 Lancement du backend avec uvicorn..."
cd "$BACKEND_DIR"
exec uvicorn api_infer:app --reload --port $PORT

