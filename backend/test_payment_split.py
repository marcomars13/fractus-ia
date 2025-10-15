import json, datetime, os

# ---- ADRESSES FOURNIES ----
WALLETS = {
    "main": "Ai5a3hEMApgb1cMaZE37nLMDSzEL56yF6XJ1VWo1HUYj",      # wallet principal (SOL) - créateur+dev
    "pool": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje",      # wallet pool/paiements (SOL/FTS)
    "treasury": "AujPwkfvposKUN31mMZMKa8tQrPcyRGdLHnuFHhoVGje"   # pour l’instant même adresse
}

# ---- RÉPARTITION (1 FTS par analyse) ----
DISTRIBUTION = { "main": 0.50, "pool": 0.40, "treasury": 0.10 }
LEDGER_FILE = "ledger.json"

def record_payment(fts_amount=1.0, note="Fractus Ultimate (multi+skyline+vector+memory)"):
    distribution = {k: round(v * fts_amount, 6) for k, v in DISTRIBUTION.items()}
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "fts_spent": float(fts_amount),
        "distribution": distribution,
        "addresses": WALLETS,
        "result": {"score": 0.972, "note": note}
    }
    ledger = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r") as f:
                ledger = json.load(f)
        except Exception:
            ledger = []
    ledger.append(record)
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"💰 Paiement reçu : {fts_amount} FTS")
    for k,v in distribution.items():
        print(f"→ {k:8s}: {v} FTS  ->  {WALLETS[k]}")
    print(f"✅ Ledger mis à jour: {LEDGER_FILE}")

if __name__ == "__main__":
    record_payment(1.0)

