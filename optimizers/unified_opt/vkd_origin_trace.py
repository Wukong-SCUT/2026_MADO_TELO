"""Bounded, opt-in VKD origin evidence. No objective calls or random draws."""

from __future__ import annotations

import json
import os

import numpy as np


MAX_OBJECTS = 16
MAX_GENERATIONS_SCANNED = 100_000
MAX_STAGES_PER_OBJECT = 5
MAX_NORMAL_BYTES = 16_000_000
MAX_FAILURE_BYTES = 2_000_000
MAX_COMMITS_PER_OBJECT = 32

_TRACES = {}


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _key(identity):
    return (str(identity.get("problem_family", "")),
            int(identity.get("function_id", -1)),
            int(identity.get("run_seed", -1)),
            int(identity.get("agent_id", -1)))


class VKDOriginTrace:
    def __init__(self, directory):
        self.directory = str(directory)
        self.normal_path = os.path.join(self.directory, f"vkd_origin_worker_{os.getpid()}.jsonl")
        self.failure_path = os.path.join(self.directory, f"vkd_origin_failure_worker_{os.getpid()}.jsonl")
        self.objects = {}
        self.normal_bytes = 0
        self.failure_bytes = 0
        self.scanned = 0
        self.normal_closed = False
        self.failure_written = set()

    def _write(self, path, row, failure=False):
        line = json.dumps(_safe(row), ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
        size = len(line.encode("utf-8"))
        budget = MAX_FAILURE_BYTES if failure else MAX_NORMAL_BYTES
        used = self.failure_bytes if failure else self.normal_bytes
        if size + used > budget:
            if not failure:
                self.normal_closed = True
            return False
        os.makedirs(self.directory, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
        if failure:
            self.failure_bytes += size
        else:
            self.normal_bytes += size
        return True

    def _object(self, identity):
        key = _key(identity)
        if key not in self.objects:
            if len(self.objects) >= MAX_OBJECTS:
                return None
            self.objects[key] = {"identity": dict(identity), "previous": None,
                                 "pending": None, "stages": set(), "commits": [],
                                 "touched": False, "boundary_sigma": None,
                                 "last_commit": None, "commit_budget": 0,
                                 "recovery_confirmed": False}
        return self.objects[key]

    def wants_generation(self, identity):
        if self.normal_closed or self.scanned >= MAX_GENERATIONS_SCANNED:
            return False
        obj = self._object(identity)
        return obj is not None and (
            len(obj["stages"]) < MAX_STAGES_PER_OBJECT or obj["pending"] is not None
        )

    def note_generation(self, identity, scalars, build_payload):
        obj = self._object(identity)
        if obj is None:
            return
        self.scanned += 1
        if self.normal_closed or self.scanned > MAX_GENERATIONS_SCANNED:
            return
        prior = obj["previous"]
        stages = obj["stages"]
        candidate_oob = int(scalars["clip_coordinates"]) > 0
        center_oob = bool(scalars["center_oob"])
        grew = float(scalars["sigma_after"]) > float(scalars["sigma_before"])
        stage = []
        if "first_positive" not in stages and scalars["alpha"] > 0 and grew:
            stage.append("first_positive")
        if "first_candidate_oob" not in stages and candidate_oob:
            stage.append("first_candidate_oob")
        if "first_center_oob" not in stages and center_oob:
            stage.append("first_center_oob")
        if ("first_score_saturation" not in stages and grew and prior is not None
              and candidate_oob and prior.get("clip_coordinates", 0) > 0
              and np.array_equal(scalars["scored_probes"], prior["scored_probes"])):
            stage.append("first_score_saturation")
        if ("recovery_control" not in stages and obj["touched"] and not candidate_oob
              and not center_oob and obj["boundary_sigma"] is not None
              and float(scalars["sigma_after"]) < obj["boundary_sigma"]):
            stage.append("recovery_control")
        if candidate_oob and not obj["touched"]:
            obj["touched"] = True
            obj["boundary_sigma"] = float(scalars["sigma_after"])
        payload_cache = None
        def current_payload():
            nonlocal payload_cache
            if payload_cache is None:
                payload_cache = build_payload()
            return payload_cache
        if obj["pending"] is not None:
            pending = obj["pending"]
            relation = ("next" if pending["env_step"] == identity.get("env_step")
                        else "next_event")
            if ("recovery_control" in pending["stages"] and relation == "next"
                    and not candidate_oob and not center_oob
                    and float(scalars["sigma_after"]) < obj["boundary_sigma"]):
                obj["recovery_confirmed"] = True
            for pending_stage in pending["stages"]:
                self._write(self.normal_path, {"type": "generation", "stage": pending_stage,
                                               "relative": relation, "data": current_payload()})
            obj["pending"] = None
        opened = []
        for selected in stage:
            if len(stages) >= MAX_STAGES_PER_OBJECT:
                break
            if prior is not None:
                previous_event = prior["payload"].get("identity", {}).get("env_step")
                relation = "previous" if previous_event == identity.get("env_step") else "prior_event"
                self._write(self.normal_path, {"type": "generation", "stage": selected,
                                               "relative": relation, "data": prior["payload"]})
            if self._write(self.normal_path, {"type": "generation", "stage": selected,
                                              "relative": "current", "data": current_payload()}):
                stages.add(selected)
                opened.append(selected)
        if opened:
            obj["pending"] = {"stages": opened, "env_step": identity.get("env_step")}
            if obj["last_commit"] is not None:
                self._write(self.normal_path, {"type": "preceding_commit",
                                               "stages": opened, "data": obj["last_commit"]})
            obj["commit_budget"] = min(3, obj["commit_budget"] + 2)
        if len(stages) < MAX_STAGES_PER_OBJECT:
            obj["previous"] = {"payload": current_payload(),
                               "clip_coordinates": int(scalars["clip_coordinates"]),
                               "scored_probes": np.asarray(scalars["scored_probes"]).copy()}
        else:
            obj["previous"] = None

    def note_commit(self, identity, slot_id, before, after, session):
        obj = self._object(identity)
        if obj is None:
            return
        row = {"type": "commit", "identity": identity, "slot_id": int(slot_id),
               "center_before": np.asarray(before), "center_after": np.asarray(after),
               "sigma": session.evolved_sigma()}
        if obj["commit_budget"] > 0 and not self.normal_closed:
            row["state_after"] = session.state_trace_snapshot()
        obj["last_commit"] = row
        obj["commits"].append(row)
        if len(obj["commits"]) > MAX_COMMITS_PER_OBJECT:
            del obj["commits"][0]
        if obj["commit_budget"] > 0 and not self.normal_closed:
            self._write(self.normal_path, row)
            obj["commit_budget"] -= 1

    def failure(self, identity, field, slot_id, agent_id, bad_indices, session=None):
        key = _key(identity)
        marker = (key, str(field), int(slot_id), int(agent_id))
        if marker in self.failure_written:
            return False
        obj = self._object(identity)
        row = {"type": "failure", "identity": identity, "field": str(field),
               "slot_id": int(slot_id), "agent_id": int(agent_id),
               "bad_indices": bad_indices,
               "state": session.state_trace_snapshot() if session is not None else None,
               "last_commit": None if obj is None else obj["last_commit"],
               "recent_commits": [] if obj is None else obj["commits"],
               "stages": [] if obj is None else sorted(obj["stages"]),
               "recovery_control_present": any(
                   item["recovery_confirmed"]
                   for other_key, item in self.objects.items() if other_key != key)}
        written = self._write(self.failure_path, row, failure=True)
        if written:
            self.failure_written.add(marker)
        return written


def get_trace(directory):
    directory = str(directory or "").strip()
    if not directory:
        return None
    trace = _TRACES.get(directory)
    if trace is None:
        trace = VKDOriginTrace(directory)
        _TRACES[directory] = trace
    return trace


def failure_from_matrix(directory, identity_base, field, value):
    trace = get_trace(directory)
    if trace is None:
        return
    bad = np.argwhere(~np.isfinite(value))
    for index in bad[:16]:
        slot_id = int(index[0]) if value.ndim == 2 else -1
        agent_id = int(index[-1])
        identity = dict(identity_base, agent_id=agent_id)
        trace.failure(identity, field, slot_id, agent_id, index.tolist())
