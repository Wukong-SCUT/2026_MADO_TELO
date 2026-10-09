"""Opt-in, best-effort numeric telemetry shared by CMAES variants."""

from collections import defaultdict
import json
import os
import time
from typing import Any, Dict

import numpy as np


COUNTER_AUDIT_MAX_EVENTS = 2048
_COUNTER_AUDIT_STATE = {}


def array_summary(name: str, value) -> Dict[str, Any]:
    """Return warning-free finite/extrema diagnostics for a numeric array."""
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        f"{name}_shape": [int(x) for x in arr.shape],
        f"{name}_all_finite": bool(finite.size == arr.size),
        f"{name}_finite_count": int(finite.size),
        f"{name}_min": float(np.min(finite)) if finite.size else None,
        f"{name}_max": float(np.max(finite)) if finite.size else None,
    }


def _json_safe(value):
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class OptimizerNumericTelemetry:
    """Collect small generation records and optionally append them to JSONL."""

    def __init__(self, optimizer_name: str, options: Dict):
        self.optimizer_name = str(optimizer_name)
        self.enabled = bool(options.get("optimizer_numeric_telemetry_enable", False))
        self.counters_enabled = bool(
            options.get("optimizer_numeric_counter_enable", self.enabled)
        )
        self.counter_audit_dir = str(
            options.get("optimizer_numeric_counter_dir", "")
        ).strip()
        self.output_dir = str(options.get("optimizer_numeric_telemetry_dir", "")).strip()
        raw_context = options.get("optimizer_numeric_telemetry_context", {})
        self.context = dict(raw_context) if isinstance(raw_context, dict) else {}
        self.records = []
        self.counters = defaultdict(int)
        if self.counters_enabled and self.counter_audit_dir:
            self._audit_counter("enabled")

    def _audit_counter(self, event: str, amount: int = 0) -> None:
        key = (os.getpid(), self.counter_audit_dir)
        state = _COUNTER_AUDIT_STATE.setdefault(key, {"events": 0, "enabled": False, "truncated": False})
        if event == "enabled":
            if state["enabled"]:
                return
            state["enabled"] = True
        elif state["events"] >= COUNTER_AUDIT_MAX_EVENTS:
            if state["truncated"]:
                return
            state["truncated"] = True
            event = "truncated"
        else:
            state["events"] += 1
        payload = {
            "event": event,
            "amount": int(amount),
            "pid": int(os.getpid()),
            "optimizer": self.optimizer_name,
            "context": self.context,
            "event_limit": COUNTER_AUDIT_MAX_EVENTS,
        }
        try:
            os.makedirs(self.counter_audit_dir, exist_ok=True)
            path = os.path.join(self.counter_audit_dir, f"numeric_counters_worker_{os.getpid()}.jsonl")
            line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except Exception as exc:
            print(f"[NumericCounterAudit] write_failed pid={os.getpid()} event={event}: {exc}", flush=True)

    def count(self, name: str, amount: int = 1) -> None:
        if self.counters_enabled:
            self.counters[str(name)] += int(amount)
            if self.counter_audit_dir:
                self._audit_counter(str(name), amount)

    def emit(self, event: str, generation: int, **fields) -> None:
        if not self.enabled:
            return
        payload = {
            "event": str(event),
            "time_unix": float(time.time()),
            "pid": int(os.getpid()),
            "optimizer": self.optimizer_name,
            "generation": int(generation),
            **self.context,
            **fields,
            "guard_counters": dict(self.counters),
        }
        payload = _json_safe(payload)
        self.records.append(payload)
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(
                self.output_dir,
                f"numeric_telemetry_worker_{os.getpid()}.jsonl",
            )
            line = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8", errors="replace")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except Exception:
            # Observability must never alter optimizer behavior or mask a failure.
            return

    def result_records(self):
        return list(self.records) if self.enabled else []

    def result_counters(self):
        return dict(self.counters) if self.counters_enabled else {}
