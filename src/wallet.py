from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import csv

import requests


DEFAULT_WALLET_API_URL = "http://127.0.0.1:8001/wallet"
DEFAULT_WALLET_CSV_PATH = Path("wallet.csv")


@dataclass
class Wallet:
    positions: List[Dict[str, Any]]
    source: str  # "api" | "csv" | "empty"


def _read_wallet_csv(path: Path) -> Wallet:
    if not path.exists():
        return Wallet(positions=[], source="empty")

    positions = []
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

    return Wallet(positions=positions, source="csv")


def load_wallet(api_url: str = DEFAULT_WALLET_API_URL, csv_path: Path = DEFAULT_WALLET_CSV_PATH) -> Wallet:
    # 1) Try API
    try:
        r = requests.get(api_url, timeout=1.0)
        r.raise_for_status()
        data = r.json()
        positions = data.get("positions", [])
        if isinstance(positions, list):
            # light validation
            cleaned = []
            for p in positions:
                sym = str(p.get("symbol", "")).strip()
                qty = p.get("quantity", 0)
                try:
                    qty = float(qty)
                except Exception:
                    continue
                if sym:
                    cleaned.append({"symbol": sym, "quantity": qty})
            return Wallet(positions=cleaned, source="api")
    except Exception:
        pass

    # 2) Fallback CSV
    return _read_wallet_csv(csv_path)