from fastapi import APIRouter, Query
import json, datetime, os

router = APIRouter()

# ---- CONFIG ----
WALLETS = {
    "main": "Ai5a3hEMApgb1cMaZE37nLMDSzEL56yF6XJ1VWo1HUYj",
    "pool": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje",
    "treasury": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje"
}
DISTRIBUTION = { "main": 0.50, "pool": 0.40, "treasury": 0.10 }
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

@router.post("/api/payment_split")
def payment_split(fts_amount: float = Query(1.0, ge=0.000001)):
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

