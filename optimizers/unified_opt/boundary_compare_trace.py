"""Small opt-in event-slot boundary comparison trace."""

import json
import os

import numpy as np


_TRACES = {}


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    return value


class BoundaryCompareTrace:
    MAX_OBJECTS = 16
    MAX_GENERATIONS = 2048
    MAX_SUMMARY_BYTES = 4_000_000
    MAX_KEY_BYTES = 2_000_000

    def __init__(self, directory):
        self.directory = str(directory)
        self.summary_path = os.path.join(self.directory, f"boundary_summary_worker_{os.getpid()}.jsonl")
        self.key_path = os.path.join(self.directory, f"boundary_key_worker_{os.getpid()}.jsonl")
        self.objects = {}
        self.summary_bytes = 0
        self.key_bytes = 0
        self.generations = 0
        self.truncations = set()

    def _mark_truncated(self, reason):
        if reason in self.truncations:
            return
        self.truncations.add(reason)
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"boundary_status_worker_{os.getpid()}.json")
        with open(path, "w", encoding="ascii") as stream:
            json.dump({"truncated": sorted(self.truncations)}, stream)

    def _write(self, path, row, limit, used):
        encoded = (json.dumps(_safe(row), ensure_ascii=True, allow_nan=False,
                              separators=(",", ":")) + "\n").encode("ascii")
        if used + len(encoded) > limit:
            return 0
        os.makedirs(self.directory, exist_ok=True)
        with open(path, "ab") as stream:
            stream.write(encoded)
        return len(encoded)

    def wants_key(self, identity):
        if self.key_bytes >= self.MAX_KEY_BYTES or self.summary_bytes >= self.MAX_SUMMARY_BYTES:
            return False
        key = (int(identity["env_step"]), int(identity["agent_id"]))
        obj = self.objects.get(key)
        return (len(self.objects) < self.MAX_OBJECTS if obj is None
                else not obj["touched"] or obj["next"])

    def note_generation(self, identity, slot, before, after, raw, scored, fitness,
                        sigma_before, sigma_after, shape_scale, lower, upper,
                        extra=None, replay=None):
        if self.generations >= self.MAX_GENERATIONS:
            self._mark_truncated("generation_limit")
            return
        if self.summary_bytes >= self.MAX_SUMMARY_BYTES:
            return
        key = (int(identity["env_step"]), int(identity["agent_id"]))
        if key not in self.objects:
            if len(self.objects) >= self.MAX_OBJECTS:
                self._mark_truncated("object_limit")
                return
            self.objects[key] = {"touched": False, "previous": None, "next": False}
        obj = self.objects[key]
        self.generations += 1
        raw = np.asarray(raw, dtype=np.float64)
        scored = np.asarray(scored, dtype=np.float64)
        fitness = np.asarray(fitness, dtype=np.float64).reshape(-1)
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        if raw.shape != scored.shape or raw.shape[0] != fitness.size:
            raise ValueError("boundary trace candidates/fitness shape mismatch")
        touched = bool(np.any(raw != scored))
        distances = np.linalg.norm(scored - before, axis=1)
        raw_distances = np.linalg.norm(raw - before, axis=1)
        row = {
            "type": "generation", "identity": identity, "slot": int(slot),
            "generation": int(self.generations), "mean_before": before,
            "mean_after": after, "sigma_before": float(sigma_before),
            "sigma_after": float(sigma_after), "shape_scale_rms": float(shape_scale),
            "raw_oob_candidates": int(np.count_nonzero(np.any(raw != scored, axis=1))),
            "raw_oob_coordinates": int(np.count_nonzero(raw != scored)),
            "mean_oob_after": bool(np.any((after < lower) | (after > upper))),
            "scored_distance_rms": float(np.sqrt(np.mean(np.square(distances)))),
            "scored_distance_max": float(np.max(distances)),
            "raw_distance_rms": float(np.sqrt(np.mean(np.square(raw_distances)))),
            "raw_distance_max": float(np.max(raw_distances)),
            "fitness_min": float(np.min(fitness)), "extra": extra or {},
        }
        size = self._write(self.summary_path, row, self.MAX_SUMMARY_BYTES, self.summary_bytes)
        if not size:
            self.summary_bytes = self.MAX_SUMMARY_BYTES
            self._mark_truncated("summary_bytes")
            obj["previous"] = None
            return
        self.summary_bytes += size
        if self.key_bytes >= self.MAX_KEY_BYTES or (obj["touched"] and not obj["next"]):
            obj["previous"] = None
            return
        payload = dict(row, raw_candidates=raw, scored_candidates=scored,
                       fitness=fitness, ranking=np.argsort(fitness), replay=replay)
        if touched and not obj["touched"]:
            for stage, item in (("previous", obj["previous"]), ("first_touch", payload)):
                if item is not None:
                    size = self._write(self.key_path, dict(stage=stage, data=item),
                                       self.MAX_KEY_BYTES, self.key_bytes)
                    if size:
                        self.key_bytes += size
                    else:
                        self.key_bytes = self.MAX_KEY_BYTES
                        self._mark_truncated("key_bytes")
            obj["touched"] = True
            obj["next"] = True
            obj["previous"] = None
        elif obj["next"]:
            size = self._write(self.key_path, dict(stage="next", data=payload),
                               self.MAX_KEY_BYTES, self.key_bytes)
            if size:
                self.key_bytes += size
            else:
                self.key_bytes = self.MAX_KEY_BYTES
                self._mark_truncated("key_bytes")
            obj["next"] = False
        else:
            obj["previous"] = payload

    def note_commit(self, identity, slot, internal_mean, proposal, committed, sigma,
                    transition):
        if self.summary_bytes >= self.MAX_SUMMARY_BYTES:
            return
        row = {"type": "commit", "identity": identity, "slot": int(slot),
               "internal_mean": internal_mean, "proposal": proposal,
               "committed_mean": committed, "sigma": float(sigma),
               "transition": transition}
        size = self._write(self.summary_path, row, self.MAX_SUMMARY_BYTES,
                           self.summary_bytes)
        self.summary_bytes = self.summary_bytes + size if size else self.MAX_SUMMARY_BYTES
        if not size:
            self._mark_truncated("summary_bytes")


def trace_for(directory):
    directory = os.path.abspath(str(directory))
    if directory not in _TRACES:
        _TRACES[directory] = BoundaryCompareTrace(directory)
    return _TRACES[directory]
