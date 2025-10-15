# server.py
import os, json, time, sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import jwt

# ====== ENV ======
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "REPLACE_WITH_YOUR_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_SUPER_SECRET_256_BITS")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

FTS_MINT = "BMAT7DrBHBEqxm8vxUwcYaKjKBdvGEM2uLH3XuztFawy"
RECEIVER_WALLET = "Ai5a3hEMApgb1cMaZE37nLMDSzEL56yF6XJ1VWo1HUYj"  # destinataire FTS
FTS_DECIMALS = 6  # FTS a 6 décimales d’après ton code

# ====== DB ======
DB_PATH = os.path.expanduser("~/Projets/fractus-ia/backend/licenses.db")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS licenses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wallet TEXT NOT NULL,
        license_type TEXT NOT NULL,
        tx_sig TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        status TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS license_configs(
        license_type TEXT PRIMARY KEY,
        duration_hours INTEGER NOT NULL
    );
    """)
    # valeurs par défaut si absentes
    existing = conn.execute("SELECT COUNT(*) AS c FROM license_configs").fetchone()["c"]
    if existing == 0:
        conn.executemany(
            "INSERT OR REPLACE INTO license_configs(license_type, duration_hours) VALUES(?,?)",
            [
                ("trial", 24),
                ("research", 30*24),
                ("pro", 365*24),
            ]
        )
    conn.commit()
    conn.close()

init_db()

# ====== APP ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Models ======
class ConfirmPayload(BaseModel):
    wallet: str
    tx_sig: str
    license_type: str  # "trial" | "research" | "pro"


# ====== Utils ======
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def add_hours(hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

def get_license_hours(license_type: str) -> int:
    conn = db()
    row = conn.execute(
        "SELECT duration_hours FROM license_configs WHERE license_type=?",
        (license_type.lower(),)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=400, detail="Unknown license_type")
    return int(row["duration_hours"])

def helius_rpc(method: str, params: list) -> Dict[str, Any]:
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    r = requests.post(url, json=body, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Helius RPC error: {r.text}")
    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"Helius RPC error: {data['error']}")
    return data["result"]

def validate_fts_transfer(tx_sig: str, from_wallet: str, to_wallet: str, expected_ui_amount: float) -> bool:
    """
    Valide qu'une transaction SPL a bien transféré 'expected_ui_amount' de FTS (mint FTS_MINT)
    de from_wallet -> to_wallet. On s'appuie sur getTransaction jsonParsed.
    """
    result = helius_rpc(
        "getTransaction",
        [tx_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
    )
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    meta = result.get("meta")
    if not meta or meta.get("err") is not None:
        raise HTTPException(status_code=400, detail="Transaction failed on-chain")

    pre_bal = meta.get("preTokenBalances", []) or []
    post_bal = meta.get("postTokenBalances", []) or []
    # On mappe: (owner, mint) -> (uiAmount, decimals)
    def balances_map(bals):
        m = {}
        for b in bals:
            owner = b.get("owner")
            mint = b.get("mint")
            ui = b.get("uiTokenAmount", {})
            ui_amount = float(ui.get("uiAmount", 0) or 0)
            decimals = int(ui.get("decimals", 0) or 0)
            m[(owner, mint)] = (ui_amount, decimals)
        return m

    pre = balances_map(pre_bal)
    post = balances_map(post_bal)

    # Delta côté sender (from_wallet, FTS_MINT) devrait être -expected_ui_amount
    # Delta côté receiver (to_wallet, FTS_MINT) devrait être +expected_ui_amount
    sender_pre = pre.get((from_wallet, FTS_MINT), (0.0, FTS_DECIMALS))[0]
    sender_post = post.get((from_wallet, FTS_MINT), (0.0, FTS_DECIMALS))[0]
    recv_pre = pre.get((to_wallet, FTS_MINT), (0.0, FTS_DECIMALS))[0]
    recv_post = post.get((to_wallet, FTS_MINT), (0.0, FTS_DECIMALS))[0]

    sender_delta = round(sender_post - sender_pre, 6)
    recv_delta = round(recv_post - recv_pre, 6)
    exp = round(expected_ui_amount, 6)

    # tolérance flottante minime
    if round(sender_delta, 6) == round(-exp, 6) and round(recv_delta, 6) == round(exp, 6):
        return True

    # fallback: inspecte logs "token" si besoin (parfois le owner n’est pas mappé si création ATA)
    # On essaye de lire les instructions parsées pour les transferts SPL
    tx = result.get("transaction", {})
    message = tx.get("message", {})
    instructions = message.get("instructions", []) or []
    # jsonParsed top-level (v0 tx)
    for ins in instructions:
        parsed = ins.get("parsed")
        if parsed and parsed.get("type") == "transferChecked":
            info = parsed.get("info", {})
            mint = info.get("mint")
            owner = info.get("owner")
            dst = info.get("destinationOwner") or info.get("destination")
            amt = float(info.get("tokenAmount", {}).get("uiAmount", 0) or 0)
            if mint == FTS_MINT and owner == from_wallet and amt == expected_ui_amount:
                # destinationOwner peut ne pas apparaître => on ignore la vérif stricte du to_wallet dans ce fallback
                return True

    return False

def issue_jwt(wallet: str, license_type: str, expires_at_iso: str, tx_sig: str) -> str:
    payload = {
        "wallet": wallet,
        "license_type": license_type,
        "tx": tx_sig,
        "exp": int(datetime.fromisoformat(expires_at_iso).timestamp())
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# ====== Routes ======
@app.get("/health")
def health():
    return {"ok": True, "time": now_utc_iso()}

@app.post("/api/payment/confirm")
def payment_confirm(body: ConfirmPayload):
    wallet = body.wallet.strip()
    license_type = body.license_type.lower().strip()
    tx_sig = body.tx_sig.strip()

    if license_type not in ("trial", "research", "pro"):
        raise HTTPException(status_code=400, detail="license_type must be trial|research|pro")

    # Montants attendus (en FTS uiAmount), cohérents avec ta page
    required_amounts = {
        "trial": 50.0,
        "research": 200.0,
        "pro": 1000.0
    }
    expected_amount = required_amounts[license_type]

    # 1) Valider on-chain via Helius
    ok = validate_fts_transfer(
        tx_sig=tx_sig,
        from_wallet=wallet,
        to_wallet=RECEIVER_WALLET,
        expected_ui_amount=expected_amount
    )
    if not ok:
        raise HTTPException(status_code=400, detail="On-chain validation failed for this payment.")

    # 2) Créer / mettre à jour la licence
    hours = get_license_hours(license_type)
    created = now_utc_iso()
    expires = add_hours(hours)

    conn = db()
    try:
        conn.execute("""
            INSERT INTO licenses(wallet, license_type, tx_sig, created_at, expires_at, status)
            VALUES(?,?,?,?,?,?)
        """, (wallet, license_type, tx_sig, created, expires, "active"))
        conn.commit()
    except sqlite3.IntegrityError:
        # tx déjà traité
        row = conn.execute("SELECT * FROM licenses WHERE tx_sig=?", (tx_sig,)).fetchone()
        conn.close()
        # on renvoie quand même un JWT valide pour cette licence
        token = issue_jwt(wallet=row["wallet"], license_type=row["license_type"], expires_at_iso=row["expires_at"], tx_sig=row["tx_sig"])
        return {"jwt": token, "status": "already_confirmed", "expires_at": row["expires_at"]}

    conn.close()

    # 3) JWT
    token = issue_jwt(wallet=wallet, license_type=license_type, expires_at_iso=expires, tx_sig=tx_sig)
    return {"jwt": token, "status": "active", "expires_at": expires}

@app.get("/api/licenses/{wallet}")
def get_active_license(wallet: str):
    conn = db()
    rows = conn.execute("""
        SELECT wallet, license_type, tx_sig, created_at, expires_at, status
        FROM licenses
        WHERE wallet=? AND status='active'
        ORDER BY expires_at DESC
        """, (wallet,)
    ).fetchall()
    conn.close()
    # filtre celles qui ne sont pas expirées
    active = []
    now_ts = time.time()
    for r in rows:
        exp_ts = datetime.fromisoformat(r["expires_at"]).timestamp()
        if exp_ts > now_ts:
            active.append(dict(r))
    return {"licenses": active}

