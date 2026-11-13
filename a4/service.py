"""The deployed configuration that Assignment 4 monitors.

Gatekeeper: the backend check (hmac.compare_digest) is the only authority (Assignment 3: G3). The LLM runs in
shadow mode and the injection detector runs as a monitoring signal; neither can grant or deny access.
RAG: Assignment 3 configuration R2V = source validation (M2) + provenance prompt + output verification (M3).
Every request is traced (MLflow spans) and returned as a flat record for the monitor.
"""
import hmac
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a3"))
import rag_pipeline as rp  # noqa: E402
import run_experiment as rx  # noqa: E402
import verifier as vf  # noqa: E402

VALID, GRANT, DENY = rx.VALID, rx.GRANT, rx.DENY
DET_THRESHOLD, NLI_THRESHOLD = 0.5, 0.5
LOCKDOWN_MESSAGE = "ACCESS_DENIED. Automatic access decisions are suspended; please contact the security desk for manual processing."


def clip(text, n=300):
    text = "" if text is None else str(text)
    return text if len(text) <= n else text[:n] + f"…[+{len(text) - n} chars]"


class _NoSpan:
    def set_inputs(self, *_a, **_k): pass
    def set_outputs(self, *_a, **_k): pass
    def set_attributes(self, *_a, **_k): pass


class Tracer:
    """Thin wrapper so the service runs with or without MLflow."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled:
            import mlflow
            self.mlflow = mlflow

    @contextmanager
    def span(self, name, span_type="UNKNOWN"):
        if not self.enabled:
            yield _NoSpan()
        else:
            with self.mlflow.start_span(name=name, span_type=span_type) as s:
                yield s

    def tag(self, tags, session=None):
        if self.enabled:
            self.mlflow.update_current_trace(tags={k: str(v) for k, v in tags.items()}, session_id=session)

    def last_trace_id(self):
        return self.mlflow.get_last_active_trace_id() if self.enabled else None


class Service:
    def __init__(self, gen, det, nli, corpus, tracer):
        self.gen, self.det, self.nli, self.tracer = gen, det, nli, tracer
        self.kb = {d["id"]: dict(d) for d in corpus}                      # live knowledge base
        self.signed = rp.manifest(corpus)                                  # signed manifest of VERIFIED documents
        self.approved_wiki = {d["id"]: rp.sha256(d["text"]) for d in corpus if d["tier"] == "community"}
        self.signed_source = {d["id"]: dict(d) for d in corpus}           # owners' copy, used only for restores
        self.gate_mode, self.rag_mode = "NORMAL", "NORMAL"
        self.shadow_enabled = True
        self.window, self.phase = 0, ""
        self.events = []                                                   # immediate integrity events

    # ------------------------------------------------------------------ knowledge-base governance
    def approved(self, d):
        h = rp.sha256(d["text"])
        return self.signed.get(d["id"]) == h if d["tier"] == "verified" else self.approved_wiki.get(d["id"]) == h

    def kb_diff(self):
        wiki = [d["id"] for d in self.kb.values() if d["tier"] == "community" and not self.approved(d)]
        verified = [d["id"] for d in self.kb.values() if d["tier"] == "verified" and not self.approved(d)]
        return {"community_unapproved": sorted(wiki), "verified_mismatch": sorted(verified)}

    def restore_wiki(self):
        """Human decision (simulated in the experiment): reject pending wiki changes, restore the approved snapshot."""
        for doc_id in [i for i, d in self.kb.items() if d["tier"] == "community" and not self.approved(d)]:
            if doc_id in self.signed_source:
                self.kb[doc_id] = dict(self.signed_source[doc_id])
            else:
                del self.kb[doc_id]

    def restore_verified(self):
        for doc_id in [i for i, d in self.kb.items() if d["tier"] == "verified" and not self.approved(d)]:
            self.kb[doc_id] = dict(self.signed_source[doc_id])

    # ------------------------------------------------------------------ request handlers
    def handle(self, req, source="user"):
        return self.handle_gate(req, source) if req["system"] == "gate" else self.handle_rag(req, source)

    def handle_gate(self, req, source):
        code, msg = req["supplied_code"], req["user_message"]
        shown = req.get("input_label") or clip(msg)          # third-party attack texts are not written to logs
        rec = {"system": "gate", "source": source, "mode": self.gate_mode, "error": None, "input_shown": shown}
        t0 = time.perf_counter()
        with self.tracer.span("gate_request", "CHAIN") as root:
            root.set_inputs({"supplied_code": clip(code, 40), "user_message": shown})
            err = rx.app_input_check(code, msg)
            backend_ok = err is None and hmac.compare_digest(code.encode("utf-8"), VALID.encode("utf-8"))
            if self.gate_mode == "LOCKDOWN":
                decision, delivered = DENY, LOCKDOWN_MESSAGE
            else:
                decision = GRANT if backend_ok else DENY
                delivered = decision
            score, det_error = None, None
            with self.tracer.span("injection_detector", "GUARDRAIL") as s:
                s.set_inputs({"text": shown})
                try:
                    score = float(self.det.scan(msg)["score_windowed"])
                except Exception as e:  # monitoring signal: failure is recorded, never blocks the decision
                    det_error = repr(e)[:120]
                s.set_outputs({"score": score, "flag": bool(score is not None and score >= DET_THRESHOLD), "error": det_error})
            shadow_raw, shadow_error = None, None
            if self.shadow_enabled and err is None:
                with self.tracer.span("shadow_llm", "LLM") as s:
                    s.set_inputs({"role": "shadow gatekeeper (no authority)", "user_message": shown})
                    try:
                        shadow_raw = self.gen.generate(rx.gate_messages(code, msg))["raw"]
                    except ValueError as e:
                        shadow_error = str(e)
                    s.set_outputs({"raw": clip(shadow_raw, 120), "error": shadow_error})
            shadow_grant = bool(shadow_raw) and shadow_raw.strip() == GRANT
            det_flag = bool(score is not None and score >= DET_THRESHOLD)
            root.set_outputs({"decision": decision, "delivered": clip(delivered, 140)})
            root.set_attributes({"backend_ok": backend_ok, "shadow_grant": shadow_grant, "detector_flag": det_flag})
            self.tracer.tag({"window": f"W{self.window:02d}", "phase": self.phase, "system": "gate", "source": source,
                             "decision": decision, "shadow_grant": shadow_grant, "detector_flag": det_flag,
                             "would_be_unauthorized_grant": shadow_grant and decision == DENY}, session=f"W{self.window:02d}")
            proc_ms = 1000 * (time.perf_counter() - t0)
        rec.update(decision=decision, backend_ok=backend_ok, app_error=err, det_score=score, det_flag=det_flag, det_error=det_error,
                   shadow_raw=shadow_raw, shadow_grant=shadow_grant, shadow_error=shadow_error,
                   latency_ms=proc_ms, trace_export_ms=1000 * (time.perf_counter() - t0) - proc_ms, trace_id=self.tracer.last_trace_id())
        return rec

    def handle_rag(self, req, source):
        query = req["query"]
        rec = {"system": "rag", "source": source, "mode": self.rag_mode, "qid": req.get("qid"), "error": None}
        t0 = time.perf_counter()
        with self.tracer.span("rag_request", "CHAIN") as root:
            root.set_inputs({"question": clip(query)})
            docs = list(self.kb.values())
            if self.rag_mode == "SAFE":
                docs = [d for d in docs if self.approved(d)]
            with self.tracer.span("retrieve_bm25", "RETRIEVER") as s:
                s.set_inputs({"query": clip(query), "index_size": len(docs), "mode": self.rag_mode})
                retrieved = rp.bm25(query, docs)
                s.set_outputs([{"id": d["id"], "tier": d["tier"], "score": d["score"]} for d in retrieved])
            with self.tracer.span("source_validation", "GUARDRAIL") as s:
                kept, quarantined = rp.validate_sources(retrieved, self.signed)
                s.set_outputs({"kept": [f"{d['id']}:{d['trust']}" for d in kept], "quarantined": quarantined})
            integrity = [q["id"] for q in quarantined if q["reason"] == "integrity_hash_mismatch"]
            for doc_id in integrity:
                self.events.append({"type": "integrity_hash_mismatch", "doc": doc_id})
            raw, status, final, issues, gen_error = None, "error", "", [], None
            try:
                with self.tracer.span("generate", "LLM") as s:
                    s.set_inputs({"question": clip(query), "context_ids": [d["id"] for d in kept]})
                    raw = self.gen.generate(rp.messages(query, kept, rp.PROVENANCE_SYSTEM, labelled=True))["raw"]
                    s.set_outputs({"raw": clip(raw, 400)})
                with self.tracer.span("output_verification", "GUARDRAIL") as s:
                    top = retrieved[0]["score"] if retrieved else 0.0
                    v = vf.verify(raw, kept, self.nli, NLI_THRESHOLD, min_score=0.5 * top)
                    status, final = v["status"], v["final"]
                    issues = sorted({i.split(":")[0] for c in v["checks"] for i in c["issues"]})
                    s.set_outputs({"status": status, "issues": issues})
            except ValueError as e:
                gen_error = str(e)
                final = "The assistant could not process this request. Please contact HR or the IT service desk."
            root.set_outputs({"answer": clip(final, 400)})
            root.set_attributes({"verify_status": status, "quarantined": integrity})
            self.tracer.tag({"window": f"W{self.window:02d}", "phase": self.phase, "system": "rag", "source": source,
                             "mode": self.rag_mode, "verify_status": status, "integrity_quarantine": bool(integrity)},
                            session=f"W{self.window:02d}")
            proc_ms = 1000 * (time.perf_counter() - t0)
        rec.update(query=query, retrieved=[d["id"] for d in retrieved], context=[d["id"] for d in kept], quarantined=integrity,
                   raw=raw, final=final, verify_status=status, issues=issues, error=gen_error,
                   latency_ms=proc_ms, trace_export_ms=1000 * (time.perf_counter() - t0) - proc_ms, trace_id=self.tracer.last_trace_id())
        return rec
