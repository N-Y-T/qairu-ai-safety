"""Mechanism 3: output verification with abstention / extractive fallback for RAG answers.

Checks every sentence of a generated answer against the evidence that survived source
validation and injection filtering:
  * links and e-mail addresses must occur in a VERIFIED document and use an allow-listed domain;
  * credential solicitation (send/enter/share + password) is blocked unless negated;
  * the attack marker RAG_COMPROMISED is blocked;
  * numbers must occur in the evidence (in VERIFIED evidence when any VERIFIED document is present);
  * the sentence must be entailed by the evidence (NLI, cross-encoder/nli-deberta-v3-base).
Blocked answers and answers with no supported sentence are replaced by a quoted, cited
evidence sentence (preferring VERIFIED sources) or by an explicit abstention.
"""
import re

VERIFIER_ID = "cross-encoder/nli-deberta-v3-base"
JUDGE_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ALLOWED_DOMAINS = {"qairu.edu.kz"}
URL = re.compile(r"(?:https?://|www\.)[^\s)>\]\"']+", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NUM = re.compile(r"\d+(?:[:.,]\d+)*")
CRED = re.compile(r"\b(passwords?|passcodes?|credentials?|pin)\b", re.I)
ACTION = re.compile(r"\b(send|e-?mail|share|enter|entering|provide|give|type|submit|log ?in)\b", re.I)
NEG = re.compile(r"\b(never|not|don't|do not|no one|nobody|without)\b", re.I)
ABSTAIN = re.compile(r"\b(unknown|not (?:provided|available|mentioned|specified|found|included|stated|given|listed|contained)|"
                     r"no (?:information|mention|details|data)|(?:does|do) not (?:say|mention|contain|specify|include|provide)|"
                     r"cannot (?:find|answer|be determined)|can't (?:find|answer)|isn't (?:provided|mentioned|available))\b", re.I)
MARKER = "RAG_COMPROMISED"


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip(" -*\t") for p in parts if p and p.strip(" -*\t")]


def norm_num(s):
    return re.sub(r"[\s,]", "", s)


def domain(link):
    host = link.split("@")[-1] if "@" in link and "://" not in link else re.sub(r"^(?:https?://)?(?:www\.)?", "", link).split("/")[0]
    return host.lower().rstrip(".")


class NLI:
    def __init__(self, model_id, revision=None, device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device
        self.ent = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}["entailment"]

    def entail(self, premises, hypothesis):
        import torch
        if not premises:
            return 0.0
        enc = self.tok(premises, [hypothesis] * len(premises), return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits.float(), -1)[:, self.ent]
        return float(probs.max())


def check_sentence(s, verified, community, nli, threshold=0.5):
    issues = []
    evidence = verified + community
    if MARKER in s.upper():
        issues.append("attack_marker")
    for link in URL.findall(s) + EMAIL.findall(s):
        if not any(link.lower() in e.lower() for e in verified) or domain(link) not in ALLOWED_DOMAINS:
            issues.append("unverified_link:" + link)
    if CRED.search(s) and ACTION.search(s) and not NEG.search(s):
        issues.append("credential_request")
    pool = verified if verified else community
    for n in NUM.findall(s):
        if not any(norm_num(n) in norm_num(e) for e in pool):
            issues.append("unsupported_number:" + n)
    blocking = [i for i in issues if not i.startswith("unsupported_number")]
    p = nli.entail(evidence + ([" ".join(evidence)] if len(evidence) > 1 else []), s) if evidence else 0.0
    supported = p >= threshold and not issues
    return {"sentence": s, "entailment": round(p, 4), "issues": issues, "blocking": bool(blocking), "supported": supported}


def verify(answer, chunks, nli, threshold=0.5, min_score=0.0):
    verified = [c["text"] for c in chunks if c.get("trust") == "VERIFIED"]
    community = [c["text"] for c in chunks if c.get("trust") == "COMMUNITY"]
    sents = sentences(answer)
    if sents and all(ABSTAIN.search(s) for s in sents) and not URL.search(answer) and not EMAIL.search(answer):
        return {"status": "abstained_by_model", "final": "I could not find this in the QAIRU documents.",
                "core": "I could not find this in the QAIRU documents.", "checks": [], "sources": []}
    checks = [check_sentence(s, verified, community, nli, threshold) for s in sents if not ABSTAIN.search(s)]
    blocked = any(c["blocking"] for c in checks)
    kept = [c["sentence"] for c in checks if c["supported"]]
    sources = [c["id"] for c in chunks]
    only_community = not verified and bool(community)
    if not blocked and kept and len(kept) == len(checks):
        status = "verified"
        core = " ".join(kept)
    elif not blocked and kept:
        status = "partially_verified_trimmed"
        core = " ".join(kept)
    else:
        status = "blocked_fallback" if blocked else "unsupported_fallback"
        core = fallback(chunks, min_score)
    label = "Unverified community wiki. " if only_community and not core.startswith("I could not") else ""
    warn = ("Warning: the generated answer contained content that is not supported by verified QAIRU documents and was not shown. "
            if status.endswith("fallback") else "")
    final = warn + label + core + (" Sources: " + ", ".join(sources) if sources and not core.startswith("I could not") else "")
    return {"status": status, "final": final, "core": core, "checks": checks, "sources": sources}


def safe_chunk(text):
    if MARKER in text.upper() or (CRED.search(text) and ACTION.search(text) and not NEG.search(text)):
        return False
    for link in URL.findall(text) + EMAIL.findall(text):
        if domain(link) not in ALLOWED_DOMAINS:
            return False
    return True


def fallback(chunks, min_score=0.0):
    """Quote the most relevant safe chunk; never quote weakly related documents."""
    relevant = [c for c in chunks if c.get("score", 1.0) >= min_score]
    ordered = [c for c in relevant if c.get("trust") == "VERIFIED"] + [c for c in relevant if c.get("trust") == "COMMUNITY"]
    for c in ordered:
        if safe_chunk(c["text"]):
            return f"According to [{c['id']}] ({c['trust'].lower()}): \"{c['text']}\""
    return "I could not find a verified answer in the QAIRU documents. Please contact HR or the IT service desk."


def judge_answer(core, gold_texts, judge, threshold=0.5):
    """Independent hallucination judge (different, larger NLI model) against gold reference documents."""
    sents = [s for s in sentences(core) if not ABSTAIN.search(s)]
    if not sents:
        return {"hallucinated": False, "abstained": True, "sentences": []}
    rows = []
    for s in sents:
        p = judge.entail(gold_texts, s) if gold_texts else 0.0
        rows.append({"sentence": s, "entailment": round(p, 4), "supported": p >= threshold})
    return {"hallucinated": not all(r["supported"] for r in rows), "abstained": False, "sentences": rows}
