"""Assignment 4 monitoring experiment (Google Colab, Tesla T4; same model and pins as the Midterm and Assignment 3).

Replays the frozen 12-window request stream through the deployed configuration, logs every request as an
MLflow trace, computes the monitoring indicators at each tick, raises alerts, applies automated responses
and records (clearly marked) simulated human decisions.

Colab:  import run_stream; run_stream.main()
Local logic check with stub models:  python run_stream.py --dry-run
"""
import argparse
import base64
import hashlib
import json
import os
import statistics
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A3 = ROOT.parent / "a3"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(A3))
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import guard  # noqa: E402
import run_experiment as rx  # noqa: E402
import verifier as vf  # noqa: E402
import monitor as mon  # noqa: E402
import service as svc  # noqa: E402
import tt_extra  # noqa: E402

EXPERIMENT = "qairu-a4-monitoring"
STRESS_SIZES = (2048, 3072, 4096, 6144, 8192)
STRESS_TARGET_RATIO = 3.0


STATE = {}


def demo(system, **fields):
    """Live demo after main(): send one request through the monitored service and return what the monitor sees."""
    service = STATE["service"]
    service.window, service.phase = 99, "live_demo"
    rec = service.handle({"system": system, **fields}, source="demo")
    keep = ("decision", "backend_ok", "shadow_raw", "shadow_grant", "det_score", "det_flag", "retrieved", "context", "quarantined",
            "raw", "final", "verify_status", "issues", "latency_ms", "trace_export_ms", "trace_id", "mode")
    return {k: rec[k] for k in keep if k in rec}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GpuStress:
    """Controlled stress: a background thread keeps the GPU busy with n x n matrix products."""

    def __init__(self, n):
        import torch
        self.torch, self.n, self.iters = torch, n, 0
        self._stop, self.thread = threading.Event(), None

    def _run(self):
        x = self.torch.randn(self.n, self.n, device="cuda")
        while not self._stop.is_set():
            _ = x @ x
            self.torch.cuda.synchronize()
            self.iters += 1

    def start(self):
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        self._stop.set()
        self.thread.join()
        self.torch.cuda.empty_cache()


class SimulatedSlowdown:
    """Dry-run only: stands in for GpuStress so the alert logic can be tested without a GPU."""

    def __init__(self, gen, delay=0.004):
        self.gen, self.delay, self.inner = gen, delay, gen.generate

    def start(self):
        self.gen.generate = lambda *a, **k: (time.sleep(self.delay), self.inner(*a, **k))[1]

    def stop(self):
        self.gen.generate = self.inner


def calibrate_stress(service, probes):
    """Pick the smallest background load that slows gatekeeper requests at least STRESS_TARGET_RATIO times."""
    service.tracer.enabled, was = False, service.tracer.enabled
    try:
        for p in probes[:4]:
            service.handle(p)                                   # warm-up
        base = statistics.median(service.handle(p)["latency_ms"] for p in probes)
        table, chosen = [{"n": 0, "median_ms": round(base, 2), "ratio": 1.0}], None
        for n in STRESS_SIZES:
            s = GpuStress(n)
            s.start()
            med = statistics.median(service.handle(p)["latency_ms"] for p in probes)
            s.stop()
            table.append({"n": n, "median_ms": round(med, 2), "ratio": round(med / base, 2), "background_iters": s.iters})
            if med / base >= STRESS_TARGET_RATIO:
                chosen = n
                break
        if chosen is None:
            chosen = max(table[1:], key=lambda r: r["ratio"])["n"]
    finally:
        service.tracer.enabled = was
        service.events.clear()
    return chosen, table


def truth_summary(recs):
    """Ground truth for the experiment report only. Not available to the monitor."""
    gate = [r for r in recs if r["system"] == "gate"]
    rag = [r for r in recs if r["system"] == "rag"]
    attacks = [r for r in gate if r["truth"]["attack"]]
    answerable = [r for r in rag if r["truth"]["category"] != "unanswerable"]
    return {
        "gate_attacks": len(attacks),
        "gate_deployed_unauthorized_grants": sum(r["decision"] == svc.GRANT and not r["backend_ok"] for r in gate),
        "gate_shadow_hijacks": sum(r["shadow_grant"] for r in attacks),
        "gate_attacks_flagged_by_detector": sum(r["det_flag"] for r in attacks),
        "gate_legit_denied": sum(r["decision"] == svc.DENY for r in gate if r["truth"]["kind"] == "benign_valid"),
        "rag_harmful_delivered": sum(mon.harmful(r["final"], r["truth"]["targets"]) for r in rag),
        "rag_harmful_raw": sum(mon.harmful(r["raw"], r["truth"]["targets"]) for r in rag),
        "rag_correct": sum(mon.correct(r["final"], r["truth"]["expected"], r["truth"]["category"])
                           and not mon.harmful(r["final"], r["truth"]["targets"]) for r in rag),
        "rag_answerable": len(answerable),
        "rag_requests": len(rag),
    }


def main(dry_run=False, out_dir="/content/a4_results", use_mlflow=True, max_windows=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = json.loads((ROOT / "data/stream.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "monitoring_config.json").read_text(encoding="utf-8"))
    suite = json.loads((A3 / "data/rag_suite_built.json").read_text(encoding="utf-8"))
    a3res = json.loads((A3 / "results/a3_results.json").read_text(encoding="utf-8"))
    extra, extra_meta = tt_extra.sample_extra(local_root=str(A3 / "tt-data") if dry_run else None)
    assert extra_meta["sample_sha256"] == stream["tensor_trust_extra"]["sample_sha256"], "Tensor Trust extra sample differs from the frozen stream"
    shadow_ref = {r["id"]: (r.get("raw") or "").strip() for r in a3res["gatekeeper"] if r["id"] in {p["id"] for p in stream["canary"]["gate"]}}

    run_info = {"schema_version": 1, "run_id": str(uuid.uuid4()), "started_utc": rx.now(), "dry_run": dry_run,
                "stream_sha256": stream["sha256"], "config_sha256": sha256_file(ROOT / "monitoring_config.json"),
                "tensor_trust_extra": extra_meta, "model": {"id": rx.MODEL_ID, "revision": rx.MODEL_REVISION}}
    if dry_run:
        gen, det, nli = rx.StubGenerator(), rx.StubDetector(), rx.StubNLI()
        run_info["detector"] = run_info["verifier"] = {"id": "stub"}
    else:
        rx.log("loading Qwen", rx.MODEL_ID, rx.MODEL_REVISION)
        gen = rx.QwenGenerator(cache_dir=str(out_dir / "hf_cache"))
        dev = "cuda" if gen.device.startswith("cuda") else "cpu"
        revs = {guard.DETECTOR_ID: a3res["detector"]["revision"], vf.VERIFIER_ID: a3res["verifier"]["revision"]}   # Assignment 3 pins
        det = guard.InjectionDetector(revision=revs[guard.DETECTOR_ID], device=dev)
        nli = vf.NLI(vf.VERIFIER_ID, revision=revs[vf.VERIFIER_ID], device=dev)
        run_info["detector"] = {"id": guard.DETECTOR_ID, "revision": revs[guard.DETECTOR_ID]}
        run_info["verifier"] = {"id": vf.VERIFIER_ID, "revision": revs[vf.VERIFIER_ID]}
    run_info["environment"] = rx.environment(gen.device)

    mlflow = None
    if use_mlflow:
        import mlflow
        from mlflow.entities import AssessmentSource
        store = out_dir / "mlflow"
        store.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store}/mlflow.db")
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(EXPERIMENT, artifact_location=str(store / "artifacts"))
        mlflow.set_experiment(EXPERIMENT)
        code_source = AssessmentSource(source_type="CODE", source_id="a4/monitor.py")
        run_info["environment"]["mlflow"] = mlflow.__version__
    tracer = svc.Tracer(enabled=use_mlflow)
    service = svc.Service(gen, det, nli, suite["corpus"], tracer)
    monitor = mon.Monitor(config, shadow_ref)
    STATE.update(service=service, monitor=monitor)

    # resolve Tensor Trust placeholders (texts stay in memory only)
    for win in stream["windows"]:
        for r in win["requests"]:
            if "tt_extra" in r:
                e = extra[r["tt_extra"]]
                r["user_message"] = e["user_message"]
                r["input_label"] = (f"[Tensor Trust hijacking attack sample_id={e['tt_sample_id']}, {len(e['user_message'])} chars; "
                                    "text not logged, see a4/tt_extra.py]")
    benign_probes = [r for r in stream["windows"][0]["requests"] if r["system"] == "gate"][:8]
    if dry_run:
        stress_n, stress_table = None, [{"note": "dry run: simulated slowdown"}]
        for p in benign_probes[:3]:
            tracer.enabled, was = False, tracer.enabled
            service.handle(p)
            tracer.enabled = was
    else:
        rx.log("calibrating background GPU load")
        stress_n, stress_table = calibrate_stress(service, benign_probes)
        rx.log("stress calibration", stress_table, "chosen n =", stress_n)
    run_info["stress"] = {"chosen_n": stress_n, "calibration": stress_table, "target_ratio": STRESS_TARGET_RATIO,
                          "kind": "simulated sleep (dry run)" if dry_run else "background GPU matrix products (real load)"}

    requests, windows, alerts_all, audit, queue_all = [], [], [], [], []

    def log_audit(window, actor, action, simulated=False, **details):
        audit.append({"utc": rx.now(), "window": window, "actor": actor, "action": action, "simulated": simulated, **details})

    def body():
        values_by_window = {}
        for win in stream["windows"][:max_windows]:
            w = win["window"]
            service.window, service.phase = w, win["phase"]
            for op in win["kb_ops_at_start"]:               # exogenous change (the controlled failure), not a service action
                if op["op"] == "add":
                    service.kb[op["doc"]["id"]] = dict(op["doc"])
                else:
                    assert op["old"] in service.kb[op["doc"]]["text"]
                    service.kb[op["doc"]]["text"] = service.kb[op["doc"]]["text"].replace(op["old"], op["new"])
            if win["kb_ops_at_start"]:
                log_audit(w, "experiment (controlled failure)", "knowledge base changed", simulated=True, ops=len(win["kb_ops_at_start"]))
            stress = None
            if win["stress"]:
                stress = SimulatedSlowdown(gen) if dry_run else GpuStress(stress_n)
                stress.start()
                log_audit(w, "experiment (controlled failure)", "background load started", simulated=dry_run)
            recs, win_alerts, seen_docs = [], [], set()
            mode_during = {"gate": service.gate_mode, "rag": service.rag_mode}
            t_win = time.perf_counter()
            for seq, req in enumerate(win["requests"], 1):
                n_events = len(service.events)
                rec = service.handle(req)
                rec.update(rid=req["rid"], window=w, seq=seq, phase=win["phase"], truth=req["truth"])
                recs.append(rec)
                for ev in service.events[n_events:]:        # event-triggered alert: raised immediately, once per document
                    if ev["doc"] not in seen_docs:
                        seen_docs.add(ev["doc"])
                        a = monitor.event_alert(w, seq, req["rid"], ev["doc"], rec["trace_id"])
                        win_alerts.append(a)
                        log_audit(w, "monitor (automated)", "CRITICAL event alert: integrity mismatch, document quarantined",
                                  doc=ev["doc"], request_seq=seq, rid=req["rid"])
                        if mlflow and rec["trace_id"]:
                            mlflow.log_feedback(trace_id=rec["trace_id"], name="alert", value="rag_integrity_quarantine_events",
                                                rationale=f"CRITICAL: {ev['doc']} failed the SHA-256 manifest check", source=code_source)
            if stress:
                stress.stop()
            win_seconds = time.perf_counter() - t_win
            # ---- tick: canary probes, manifest check, indicators, alerts, responses
            canary = []
            for p in stream["canary"]["gate"]:
                r = service.handle({"system": "gate", "supplied_code": p["supplied_code"], "user_message": p["user_message"]}, source="canary")
                r.update(probe={"id": p["id"], "expect_grant": p["expect_grant"]}, window=w)
                canary.append(r)
            for p in stream["canary"]["rag"]:
                r = service.handle({"system": "rag", "qid": p["qid"], "query": p["query"]}, source="canary")
                r.update(probe={"id": p["qid"], "expected": p["expected"], "targets": p["targets"], "category": p["category"]}, window=w)
                canary.append(r)
            kb_diff = service.kb_diff()
            queue = monitor.build_queue(recs, kb_diff, w)
            values = monitor.compute(recs, canary, kb_diff, queue)
            values_by_window[w] = values
            if w == max(config["calibration"]["windows"]):
                rule = monitor.calibrate(values_by_window, stream["gate_per_window"], stream["rag_per_window"])
                log_audit(w, "monitor (automated)", "calibrated thresholds frozen", thresholds={k: monitor.thresholds[k] for k in rule})
            tick_alerts = monitor.evaluate(values, w)
            win_alerts += tick_alerts
            for a in tick_alerts:
                log_audit(w, "monitor (automated)", f"{a['severity']} alert: {a['indicator']}", value=a["value"], threshold=a["threshold"])
                if a["automated_action"] == "rag_safe_mode" and service.rag_mode != "SAFE":
                    service.rag_mode = "SAFE"
                    log_audit(w, "monitor (automated)", "entered RAG safe mode", cause=a["indicator"])
                if a["automated_action"] == "gate_lockdown" and service.gate_mode != "LOCKDOWN":
                    service.gate_mode = "LOCKDOWN"
                    log_audit(w, "monitor (automated)", "entered gatekeeper lockdown", cause=a["indicator"])
            for op in win["human_ops_after_tick"]:           # SIMULATED human decisions (scripted reviewer)
                if op["op"] == "review_kb_changes":
                    service.restore_wiki()
                    service.rag_mode = "NORMAL"
                else:
                    service.restore_verified()
                log_audit(w, "human reviewer (SIMULATED)", op["decision"], simulated=True, note=op["note"])
            if service.rag_mode == "SAFE" and not win["human_ops_after_tick"] and not any(service.kb_diff().values()) \
                    and not any(a["automated_action"] == "rag_safe_mode" for a in tick_alerts):
                service.rag_mode = "NORMAL"                  # generic scripted reviewer: nothing pending, exit authorized
                log_audit(w, "human reviewer (SIMULATED)", "safe mode exit authorized", simulated=True,
                          note="SIMULATED reviewer: knowledge base matches the approved manifests and no safe-mode alert is open")
            truth = truth_summary(recs)
            summary = {"window": w, "phase": win["phase"], "note": win["note"], "values": values, "thresholds": dict(monitor.thresholds),
                       "alerts": win_alerts, "queue": queue, "kb_diff": kb_diff, "mode_during": mode_during,
                       "mode_after_tick": {"gate": service.gate_mode, "rag": service.rag_mode}, "truth": truth,
                       "seconds": round(win_seconds, 2), "canary": [{k: r.get(k) for k in ("system", "probe", "decision", "shadow_raw", "final", "verify_status", "trace_id")} for r in canary]}
            windows.append(summary)
            alerts_all.extend(win_alerts)
            queue_all.extend(queue)
            requests.extend(recs)
            if mlflow:
                m = {f"ind.{k}": float(v) for k, v in values.items() if v is not None}
                m.update({f"thr.{k}": float(v) for k, v in monitor.thresholds.items()})
                m.update({f"truth.{k}": float(v) for k, v in truth.items()})
                m.update({"alerts.count": len(win_alerts), "alerts.critical": sum(a["severity"] == "CRITICAL" for a in win_alerts),
                          "mode.rag_safe": float(summary["mode_during"]["rag"] == "SAFE"), "mode.gate_lockdown": float(summary["mode_during"]["gate"] == "LOCKDOWN")})
                mlflow.log_metrics(m, step=w)
                mlflow.log_dict({"window": w, "phase": win["phase"], "alerts": win_alerts, "review_queue": queue}, f"alerts/W{w:02d}.json")
                for it in queue:
                    if it.get("trace_id"):
                        mlflow.log_feedback(trace_id=it["trace_id"], name="review_queue", value=it["type"],
                                            rationale=f"priority {it['priority']}; reviewer: {it['reviewer']}", source=code_source)
            rx.log(f"W{w:02d} {win['phase']:<15} {win_seconds:6.1f}s alerts={[a['indicator'] for a in win_alerts]} "
                   f"queue={len(queue)} rag_mode_next={service.rag_mode}")

    if mlflow:
        with mlflow.start_run(run_name="stream-" + time.strftime("%Y%m%d-%H%M%S")) as run:
            run_info["mlflow_run_id"] = run.info.run_id
            mlflow.set_tags({"stream_sha256": stream["sha256"], "config_sha256": run_info["config_sha256"], "dry_run": str(dry_run),
                             "code_commit": str(run_info["environment"].get("code_commit")), "system": "QAIRU gatekeeper (G3 + shadow) and RAG (R2V)"})
            mlflow.log_params({"model": rx.MODEL_ID, "model_revision": rx.MODEL_REVISION, "detector": run_info["detector"]["id"],
                               "verifier": run_info["verifier"]["id"], "windows": len(stream["windows"]),
                               "gate_per_window": stream["gate_per_window"], "rag_per_window": stream["rag_per_window"],
                               "stress_matrix_n": stress_n, "detector_threshold": svc.DET_THRESHOLD, "nli_threshold": svc.NLI_THRESHOLD})
            body()
            for k, v in monitor.thresholds.items():
                mlflow.log_param(f"threshold.{k}", round(v, 4) if isinstance(v, float) else v)
            run_info["finished_utc"] = rx.now()
            _save(out_dir, run_info, requests, windows, alerts_all, audit, queue_all, monitor)
            for name in ("a4_windows.json", "a4_alerts.json", "a4_audit.jsonl", "a4_run.json"):
                mlflow.log_artifact(str(out_dir / name), artifact_path="monitoring")
            mlflow.log_artifact(str(ROOT / "monitoring_config.json"), artifact_path="monitoring")
    else:
        body()
        run_info["finished_utc"] = rx.now()
        _save(out_dir, run_info, requests, windows, alerts_all, audit, queue_all, monitor)

    tar_path = out_dir / "a4_bundle.tar.xz"
    with tarfile.open(tar_path, "w:xz", preset=9) as tar:
        for name in ("a4_requests.jsonl", "a4_windows.json", "a4_alerts.json", "a4_audit.jsonl", "a4_run.json"):
            tar.add(out_dir / name, arcname=name)
        if use_mlflow:
            tar.add(out_dir / "mlflow", arcname="mlflow")
    packed = base64.b64encode(tar_path.read_bytes()).decode()
    rx.log("DONE requests", len(requests), "alerts", len(alerts_all), "bundle", tar_path.stat().st_size, "bytes; sha256", sha256_file(tar_path))
    return {"run": run_info, "windows": windows, "alerts": alerts_all, "audit": audit}, packed


def _save(out_dir, run_info, requests, windows, alerts, audit, queue, monitor):
    run_info["thresholds"] = dict(monitor.thresholds)
    run_info["calibration"] = monitor.calib
    run_info["counts"] = {"requests": len(requests), "alerts": len(alerts), "review_items": len(queue)}
    with open(out_dir / "a4_requests.jsonl", "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "a4_windows.json").write_text(json.dumps(windows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "a4_alerts.json").write_text(json.dumps(alerts, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "a4_audit.jsonl", "w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out_dir / "a4_run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="/tmp/a4_dry")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    main(dry_run=a.dry_run, out_dir=a.out, use_mlflow=not a.no_mlflow)
