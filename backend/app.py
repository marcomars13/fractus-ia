#!/usr/bin/env python3
"""
Fractus Local API — Simulation (for testing only)
Author: Marc Cedrych
Description:
  Local FastAPI backend to test Fractus Access portal.
  Supports image, DNA, and log uploads + FTS token distribution (payment_split).
"""

from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json, time, os, datetime
from PIL import Image
import io

# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = FastAPI(
    title="Fractus Local API",
    version="1.1",
    description="Local backend to simulate Fractus Access portal + FTS payments"
)

# CORS pour autoriser les requêtes depuis ton site web (localhost ou fractys.io)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROUTE DE BASE (HEALTH)
# ============================================================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Fractus Local API operational"}

# ============================================================
# ROUTE D'UPLOAD (IMAGE, DNA, LOGS)
# ============================================================

@app.post("/api/fractus_analyze")
async def fractus_analyze(
    file: UploadFile,
    mode: str = Form("image"),
    cost: float = Form(0.25),
    wallet: str = Form("LOCAL-DEMO-WALLET")
):
    """Simulation d'analyse Fractus + distribution FTS"""
    data = await file.read()
    try:
        if mode == "image":
            Image.open(io.BytesIO(data))
    except Exception:
        pass

    result = {
        "lat": round(np.random.uniform(43.1, 43.4), 6),
        "lon": round(np.random.uniform(5.3, 5.5), 6),
        "score": round(np.random.uniform(0.94, 0.99), 6),
        "note": "Fractus Ultimate (multi+skyline+vector+memory)"
    }

    distribution = {
        "pool": 0.4,
        "dev": 0.5,
        "treasury": 0.1
    }

    return {
        "success": True,
        "filename": file.filename,
        "mode": mode,
        "fts_spent": cost,
        "distribution": distribution,
        "result": result
    }

# ============================================================
# ROUTE DE PAIEMENT (FTS DISTRIBUTION)
# ============================================================

WALLETS = {
    "main": "Ai5a3hEMApgb1cMaZE37nLMDSzEL56yF6XJ1VWo1HUYj",
    "pool": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje",
    "treasury": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje"
}
DISTRIBUTION = {"main": 0.50, "pool": 0.40, "treasury": 0.10}
LEDGER_FILE = "ledger.json"

def _write_ledger(entry: dict):
    ledger = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r") as f:
                ledger = json.load(f)
        except Exception:
            ledger = []
    ledger.append(entry)
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)

@app.post("/api/payment_split")
def payment_split(fts_amount: float = Query(1.0, ge=0.000001)):
    """Simule la répartition d’un paiement FTS (main/pool/treasury)"""
    distribution = {k: round(v * fts_amount, 6) for k, v in DISTRIBUTION.items()}
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "fts_spent": float(fts_amount),
        "distribution": distribution,
        "addresses": WALLETS,
        "result": {"score": 0.972, "note": "Fractus Ultimate (multi+skyline+vector+memory)"}
    }
    _write_ledger(entry)
    return {"success": True, **entry}

# ============================================================
# ROUTE DE CONSULTATION DU LEDGER
# ============================================================

@app.get("/api/ledger")
def get_ledger():
    """Renvoie le contenu du ledger local"""
    if not os.path.exists(LEDGER_FILE):
        return {"ledger": []}
    with open(LEDGER_FILE, "r") as f:
        ledger = json.load(f)
    return {"ledger": ledger}

# ============================================================
# MAIN LOCAL
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

