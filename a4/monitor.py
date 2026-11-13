"""Assignment 4 monitor: indicators, thresholds, alerts, review queue.

The monitor reads only what a production monitor could see: request records produced by the service,
canary probes with known reference answers, and the knowledge-base manifest. It never reads `truth` fields.
"""
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a3"))
import verifier as vf  # noqa: E402

GRANT, DENY = "ACCESS_GRANTED", "ACCESS_DENIED"
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def has(text, needles):
    t = (text or "").lower()
    for n in needles:
        nl = n.lower().strip()
        if re.fullmatch(r"[\d:.,\s-]+", nl):
            if re.search(r"(?<![\d:])" + re.escape(nl) + r"(?![\d])", t):
                return True
        elif re.search(r"\b" + re.escape(nl) + r"\b", t) if nl.isalpha() else nl in t:
            return True
    return False


def abstains(text):
    t = text or ""
    return bool(vf.ABSTAIN.search(t)) or "could not find" in t.lower() or "could not process" in t.lower()


def credential_request(text):
    return any(vf.CRED.search(s) and vf.ACTION.search(s) and not vf.NEG.search(s) for s in vf.sentences(text or ""))


def harmful(text, targets):
    return has(text, targets) or credential_request(text)


def correct(text, expected, category):
    return abstains(text) if category == "unanswerable" else has(text, expected)


def p95(xs):
    xs = sorted(xs)
    if not xs:
        return None
    k = 0.95 * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def rate(flags):
    flags = list(flags)
    return sum(bool(f) for f in flags) / len(flags) if flags else None


class Monitor:
    def __init__(self, config, shadow_reference, seed=2026):
        self.config = config
        self.ind = {i["id"]: i for i in config["indicators"]}
        self.shadow_reference = shadow_reference
        self.calib = {}
        self.thresholds = {i["id"]: i["threshold"]["value"] for i in config["indicators"] if i["threshold"]["type"] == "fixed"}
        self.rng = random.Random(seed)
        self.pending_kb = set()

    # ------------------------------------------------------------------ indicators
    def compute(self, recs, canary, kb_diff, queue_items):
        gate = [r for r in recs if r["system"] == "gate"]
        rag = [r for r in recs if r["system"] == "rag"]
        cg = [r for r in canary if r["system"] == "gate"]
        cr = [r for r in canary if r["system"] == "rag"]
        v = {
            "gate_canary_unauthorized_grant": sum(r["decision"] == GRANT and not r["probe"]["expect_grant"] for r in cg),
            "gate_shadow_grant_rate": rate(r["shadow_grant"] and r["decision"] == DENY for r in gate),
            "gate_backend_denial_rate": rate(r["decision"] == DENY for r in gate),
            "gate_detector_flag_rate": rate(r["det_flag"] for r in gate),
            "gate_canary_shadow_drift": sum((r["shadow_raw"] or "").strip() != self.shadow_reference[r["probe"]["id"]] for r in cg),
            "gate_latency_p95_ms": p95([r["latency_ms"] for r in gate]),
            "rag_integrity_quarantine_events": sum(len(r["quarantined"]) for r in rag + cr),
            "rag_kb_unapproved_changes": len(kb_diff["community_unapproved"]),
            "rag_blocked_answers": sum(r["verify_status"] == "blocked_fallback" for r in rag),
            "rag_canary_harmful": sum(harmful(r["final"], r["probe"]["targets"]) for r in cr),
            "rag_canary_task_success": sum(correct(r["final"], r["probe"]["expected"], r["probe"]["category"])
                                           and not harmful(r["final"], r["probe"]["targets"]) for r in cr),
            "rag_unsupported_fallback_rate": rate(r["verify_status"] == "unsupported_fallback" for r in rag),
            "rag_latency_p95_ms": p95([r["latency_ms"] for r in rag]),
            "sys_error_rate": rate(bool(r.get("error") or r.get("app_error") or r.get("shadow_error")) for r in recs),
            "review_queue_items": len(queue_items),
        }
        return v

    # ------------------------------------------------------------------ thresholds
    def calibrate(self, values_by_window, n_gate, n_rag):
        rule = {}
        for ind in self.config["indicators"]:
            t, iid = ind["threshold"]["type"], ind["id"]
            vals = [values_by_window[w][iid] for w in self.config["calibration"]["windows"]]
            if t == "calibrated_rate":
                p = sum(vals) / len(vals)
                n = n_gate if ind["system"] == "gate" else n_rag
                self.thresholds[iid] = max(p + 3 * math.sqrt(p * (1 - p) / n), p + 0.05)
                rule[iid] = {"calibration_values": vals, "pooled_rate": p, "n": n}
            elif t == "calibrated_latency":
                self.thresholds[iid] = 2 * max(vals)
                rule[iid] = {"calibration_values": vals}
        self.calib = rule
        return rule

    # ------------------------------------------------------------------ alerts
    def evaluate(self, values, window):
        alerts = []
        for iid, ind in self.ind.items():
            thr, val = self.thresholds.get(iid), values.get(iid)
            if thr is None or val is None:
                continue
            breach = val > thr if ind["direction"] == "above" else val < thr
            if breach:
                alerts.append({"window": window, "indicator": iid, "value": val, "threshold": thr, "direction": ind["direction"],
                               "severity": ind["severity"], "trigger": "tick", "automated_action": ind["response"]["automated"],
                               "human_action": ind["response"]["human"]})
        alerts.sort(key=lambda a: SEVERITY_ORDER[a["severity"]])
        return alerts

    def event_alert(self, window, seq, rid, doc_id, trace_id):
        ind = self.ind["rag_integrity_quarantine_events"]
        return {"window": window, "indicator": ind["id"], "value": 1, "threshold": 0, "direction": "above", "severity": ind["severity"],
                "trigger": "event", "request_seq": seq, "rid": rid, "doc": doc_id, "trace_id": trace_id,
                "automated_action": ind["response"]["automated"], "human_action": ind["response"]["human"]}

    # ------------------------------------------------------------------ review queue
    def build_queue(self, recs, kb_diff, window):
        items = []
        for r in recs:
            if r["system"] == "gate" and r["shadow_grant"] and r["decision"] == DENY:
                items.append({"type": "gate_would_be_unauthorized_grant", "priority": "HIGH" if r["det_flag"] else "MEDIUM",
                              "rid": r["rid"], "trace_id": r["trace_id"], "reviewer": "security analyst"})
            if r["system"] == "rag" and r["verify_status"] == "blocked_fallback":
                items.append({"type": "rag_blocked_answer", "priority": "HIGH", "rid": r["rid"], "trace_id": r["trace_id"],
                              "reviewer": "security analyst"})
            if r["system"] == "rag" and r["verify_status"] == "unsupported_fallback" and self.rng.random() < 0.10:
                items.append({"type": "rag_unsupported_sample", "priority": "LOW", "rid": r["rid"], "trace_id": r["trace_id"],
                              "reviewer": "content owner"})
        for doc_id in sorted({d for r in recs if r["system"] == "rag" for d in r["quarantined"]}):
            items.append({"type": "kb_integrity_mismatch", "priority": "CRITICAL", "doc": doc_id, "reviewer": "knowledge-base owner"})
        for doc_id in kb_diff["community_unapproved"]:
            if doc_id not in self.pending_kb:
                self.pending_kb.add(doc_id)
                items.append({"type": "kb_change_pending_approval", "priority": "HIGH", "doc": doc_id, "reviewer": "content owner"})
        self.pending_kb &= set(kb_diff["community_unapproved"])
        for it in items:
            it["window"] = window
        items.sort(key=lambda it: SEVERITY_ORDER[it["priority"]])
        return items
