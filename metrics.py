# metrics.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def now_s() -> float:
    return time.perf_counter()


@dataclass
class RunMetrics:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at_s: float = field(default_factory=now_s)
    data: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def inc(self, key: str, by: int = 1) -> None:
        self.data[key] = int(self.data.get(key, 0)) + by

    def add_timing(self, key: str, seconds: float) -> None:
        self.data[key] = float(seconds)

    def finish(self) -> Dict[str, Any]:
        self.data["run_id"] = self.run_id
        self.data["report_total_seconds"] = now_s() - self.started_at_s
        self.data["ts_unix"] = time.time()
        return dict(self.data)


class Timer:
    def __init__(self, metrics: RunMetrics, key: str):
        self.metrics = metrics
        self.key = key
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = now_s()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.metrics.add_timing(self.key, now_s() - self.t0)


def approx_tokens_from_chars(n_chars: int) -> int:
    # simpele proxy: 4 chars ≈ 1 token
    return int(round(n_chars / 4))


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")