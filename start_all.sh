#!/bin/zsh
cd ~/Projets/fractus-ia

echo "🛑 Arrêt des anciens serveurs..."
pkill -f "uvicorn" || true
pkill -f "node" || true
sleep 2
echo "✅ Ports libérés."

echo "🚀 Démarrage backend..."
. .venv/bin/activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 &
BACK_PID=$!

echo "🚀 Démarrage frontend..."
cd frontend
npm run dev &
FRONT_PID=$!
cd ..

# Attente que le backend (8000) soit dispo
echo "⏳ Attente backend..."
until nc -z 127.0.0.1 8000; do
  sleep 1
done
echo "✅ Backend prêt."

# Attente que le frontend (3000) soit dispo
echo "⏳ Attente frontend..."
until nc -z 127.0.0.1 3000; do
  sleep 1
done
echo "✅ Frontend prêt."

# Ouvrir Swagger et Dashboard dans Safari
open "http://127.0.0.1:8000/docs"
open "http://127.0.0.1:3000/dashboard"

# Notification macOS
osascript -e 'display notification "🚀 Fractus × IA est lancé et prêt !" with title "Fractus"'

