#!/bin/bash
# 🚀 Script de lancement backend Fractus + Plonk avec nettoyage du port 8000

cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)

# Vérifie si le port 8000 est déjà utilisé
PID=$(lsof -ti:8000)
if [ -n "$PID" ]; then
  echo "⚠️  Port 8000 déjà occupé (PID=$PID). On kill le process..."
  kill -9 $PID
  sleep 1
fi

# Lancer uvicorn
echo "🚀 Démarrage du backend sur http://127.0.0.1:8000"
python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

