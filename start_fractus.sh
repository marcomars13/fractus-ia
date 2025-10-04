#!/usr/bin/env bash
# /Users/marco/Projets/fractus-ia/start_fractus.sh
set -Eeuo pipefail

ROOT="/Users/marco/Projets/fractus-ia"
BACK="$ROOT/backend"
FRONT="$ROOT/frontend"
LOGS="$ROOT/logs"

mkdir -p "$LOGS"

echo "🛑 Arrêt des anciens process uvicorn et node…"
pkill -f "uvicorn .*main:app"    || true
pkill -f "node .*next"           || true
pkill -f "next dev"              || true
sleep 0.5
pkill -9 -f "uvicorn .*main:app" || true
pkill -9 -f "node .*next"        || true
pkill -9 -f "next dev"           || true

# Backend
echo "🚀 Démarrage backend (FastAPI @127.0.0.1:8000)…"
cd "$BACK"
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi
: > "$LOGS/backend.log"
nohup uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info >> "$LOGS/backend.log" 2>&1 &
BACK_PID=$!

# Frontend
echo "🌐 Démarrage frontend (Next.js @localhost:3000)…"
cd "$FRONT"
: > "$LOGS/frontend.log"
nohup npm run dev >> "$LOGS/frontend.log" 2>&1 &
FRONT_PID=$!

echo "➡️ Backend PID: $BACK_PID → http://127.0.0.1:8000"
echo "➡️ Frontend PID: $FRONT_PID → http://localhost:3000"

# Anti-crash léger
monitor() {
  while true; do
    sleep 3
    if ! kill -0 "$BACK_PID" 2>/dev/null; then
      echo "⚠️ Backend tombé — relance…" | tee -a "$LOGS/backend.log"
      cd "$BACK"
      nohup uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info >> "$LOGS/backend.log" 2>&1 &
      BACK_PID=$!
      echo "✅ Backend relancé PID=$BACK_PID"
    fi
    if ! kill -0 "$FRONT_PID" 2>/dev/null; then
      echo "⚠️ Frontend tombé — relance…" | tee -a "$LOGS/frontend.log"
      cd "$FRONT"
      nohup npm run dev >> "$LOGS/frontend.log" 2>&1 &
      FRONT_PID=$!
      echo "✅ Frontend relancé PID=$FRONT_PID"
    fi
  done
}

monitor &

echo "📝 Logs: $LOGS/backend.log  |  $LOGS/frontend.log"
echo "✅ Fractus est lancé avec surveillance active."

