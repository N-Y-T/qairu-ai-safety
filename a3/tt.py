"""Deterministic sampling of the released Tensor Trust robustness benchmarks.

Source: HumanCompatibleAI/tensor-trust-data, commit TT_COMMIT (Toyer et al., ICLR 2024).
Only the `attack` strings are used; they are inserted as `user_message` into the
QAIRU gatekeeper with supplied_code = WRONG-CODE. The game's success phrase
"Access Granted" is mapped to this application's grant token ACCESS_GRANTED.
"""
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

TT_COMMIT = "747a75e096761ebc01bd3970158827326b4add23"
TT_FILES = {
    "hijack": "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
    "extract": "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
}
SAMPLE_SIZES = {"hijack": 200, "extract": 150}
MAX_ATTACK_CHARS = 3000
SEED = 2026


def load_rows(kind, local_root=None):
    rel = TT_FILES[kind]
    if local_root:
        text = Path(local_root, rel).read_text(encoding="utf-8")
    else:
        url = f"https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/{TT_COMMIT}/{rel}"
        text = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], hashlib.sha256(text.encode()).hexdigest()


def adapt(attack):
    return re.sub(r"access\s*granted", "ACCESS_GRANTED", attack, flags=re.IGNORECASE)


def sample(local_root=None):
    out, meta = [], {"commit": TT_COMMIT, "seed": SEED, "max_attack_chars": MAX_ATTACK_CHARS}
    for kind in ("hijack", "extract"):
        rows, digest = load_rows(kind, local_root)
        eligible = sorted([r for r in rows if len(r["attack"]) <= MAX_ATTACK_CHARS], key=lambda r: r["sample_id"])
        picked = random.Random(SEED).sample(eligible, SAMPLE_SIZES[kind])
        meta[kind] = {"file_sha256": digest, "rows": len(rows), "eligible": len(eligible), "sampled": len(picked)}
        prefix = "TTH" if kind == "hijack" else "TTE"
        for i, r in enumerate(picked):
            out.append({"id": f"{prefix}{i+1:03d}", "suite": prefix, "family": "tensor_trust_" + kind,
                        "tt_sample_id": r["sample_id"], "supplied_code": "WRONG-CODE",
                        "user_message": adapt(r["attack"]) if kind == "hijack" else r["attack"],
                        "attack": True, "goal": "hijack" if kind == "hijack" else "extract"})
    canon = json.dumps([[c["id"], c["user_message"]] for c in out], ensure_ascii=False, separators=(",", ":"))
    meta["sample_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return out, meta
