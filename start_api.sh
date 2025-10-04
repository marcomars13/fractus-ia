#!/bin/bash
echo "🛑 Arrêt des anciens serveurs sur le port 8000..."

# Tue tous les process Python/Uvicorn qui écoutent sur 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

# (Optionnel) Tue aussi les uvicorn lancés ailleurs
pkill -9 -f "uvicorn backend.main:app" 2>/dev/null

echo "✅ Port 8000 libéré."

# Active l'environnement virtuel
source .venv/bin/activate

# Relance l'API
echo "🚀 Démarrage de l'API Fractus..."
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

