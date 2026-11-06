"""Mechanism 2: prompt-injection detection with an open classifier.

Model: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0, binary SAFE/INJECTION, 512-token limit).
Two scanning modes are recorded for every text:
  * truncated  - standard pipeline usage: only the first 512 tokens are classified;
  * windowed   - our adaptation: 510-token windows with 256-token stride, max INJECTION score.
Policy is fail-closed: if the detector raises, the request is treated as flagged and
routed to human review instead of being silently allowed.
"""
import time

DETECTOR_ID = "protectai/deberta-v3-base-prompt-injection-v2"


class InjectionDetector:
    def __init__(self, model_id=DETECTOR_ID, revision=None, device="cuda", threshold=0.5,
                 window=510, stride=256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device
        self.threshold = threshold
        self.window, self.stride = window, stride
        labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
        self.inj = labels["INJECTION"]

    def _scores(self, id_windows):
        import torch
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        batch = [[cls] + w + [sep] for w in id_windows]
        width = max(map(len, batch))
        ids = torch.full((len(batch), width), self.tok.pad_token_id)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, row in enumerate(batch):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        with torch.inference_mode():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
        return torch.softmax(logits.float(), -1)[:, self.inj].tolist()

    def scan(self, text):
        start = time.perf_counter()
        ids = self.tok(text or "", add_special_tokens=False)["input_ids"]
        if not ids:
            windows = [[]]
        else:
            windows = [ids[i:i + self.window] for i in range(0, max(1, len(ids) - self.window + self.stride), self.stride)]
            if windows[-1][-1:] != ids[-1:]:
                windows.append(ids[-self.window:])
        scores = self._scores(windows)
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        return {"score_truncated": scores[0], "score_windowed": max(scores), "n_tokens": len(ids),
                "n_windows": len(windows), "seconds": time.perf_counter() - start}


def decide(scan, threshold=0.5, windowed=True, failed=False):
    """Return (flagged, reason). Fail-closed on detector errors."""
    if failed:
        return True, "detector_error_fail_closed"
    score = scan["score_windowed"] if windowed else scan["score_truncated"]
    return score >= threshold, ("injection_score_%.3f" % score)


REVIEW_MESSAGE = ("ACCESS_DENIED. Your message contains text that looks like instructions to the access system, "
                  "so it was not processed automatically and has been queued for human review (ticket {ticket}). "
                  "If this was a normal request, resend it without instructions or contact the security desk.")
