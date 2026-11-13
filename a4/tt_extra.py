"""Unseen Tensor Trust hijacking attacks for the Assignment 4 attack wave.

Samples attacks from the released benchmark that were NOT in the Assignment 3 sample
(same source commit and eligibility rule as a3/tt.py, different seed).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a3"))
import tt  # noqa: E402

EXTRA_N = 80
EXTRA_SEED = 2027


def sample_extra(local_root=None):
    rows, digest = tt.load_rows("hijack", local_root)
    eligible = sorted([r for r in rows if len(r["attack"]) <= tt.MAX_ATTACK_CHARS], key=lambda r: r["sample_id"])
    used = {r["sample_id"] for r in random.Random(tt.SEED).sample(eligible, tt.SAMPLE_SIZES["hijack"])}
    rest = [r for r in eligible if r["sample_id"] not in used]
    picked = random.Random(EXTRA_SEED).sample(rest, EXTRA_N)
    out = [{"tt_sample_id": r["sample_id"], "user_message": tt.adapt(r["attack"])} for r in picked]
    canon = json.dumps([[o["tt_sample_id"], o["user_message"]] for o in out], ensure_ascii=False, separators=(",", ":"))
    meta = {"commit": tt.TT_COMMIT, "file_sha256": digest, "eligible": len(eligible), "unused_by_a3": len(rest),
            "sampled": EXTRA_N, "seed": EXTRA_SEED, "sample_sha256": hashlib.sha256(canon.encode()).hexdigest()}
    return out, meta
