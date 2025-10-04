#!/bin/bash
# Script pour tuer les vieux uvicorn/Python et relancer l'API

# Chemin du projet
cd "$(dirname "$0")"

# Activer le venv
source .venv/bin/activate

# Port par défaut (8000 si non fourni en argument)
PORT=${1:-8000}

# Tuer tous les process sur ce port
PIDS=$(lsof -ti :$PORT)
if [ -n "$PIDS" ]; then
  echo "🛑 Killing old processes on port $PORT (PID: $PIDS)"
  kill -9 $PIDS
fi

# Relancer uvicorn
echo "🚀 Starting API on http://127.0.0.1:$PORT ..."
uvicorn api_infer:app --reload --host 127.0.0.1 --port $PORT

