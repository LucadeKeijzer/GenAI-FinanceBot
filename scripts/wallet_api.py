from fastapi import FastAPI
from datetime import datetime, timezone
import csv
from pathlib import Path

app = FastAPI(title="FinanceBot Wallet API", version="0.1")

WALLET_CSV_PATH = Path("wallet.csv")


def read_wallet_from_csv(path: Path):
    positions = []
    if not path.exists():
        return positions

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            qty_raw = (row.get("quantity") or "").strip()
            if not symbol:
                continue
            try:
                quantity = float(qty_raw)
            except ValueError:
                continue
            positions.append({"symbol": symbol, "quantity": quantity})
    return positions


@app.get("/wallet")
def get_wallet():
    positions = read_wallet_from_csv(WALLET_CSV_PATH)
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "positions": positions,
        "source": "api(csv)",
    }