"""Report figures from results/metrics.json (static PNG). Palette: validated categorical slots 1-3 + neutral inks."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 250})


def grouped(ax, groups, series, colors, labels, ymax):
    n = len(series)
    w = 0.78 / n
    for i, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, vals, w * 0.9, color=col, label=lab, edgecolor="white", linewidth=0.5)
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.012, f"{v:.0f}", ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ymax)


def main(metrics_path, outdir):
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    G = m["gatekeeper"]
    bs, lc = G["by_suite"], m["extras"]["long_context"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.6, 1]})
    groups = ["Midterm\nhijack\n(32)", "Tensor\nTrust\n(200)", "Obfuscated\nPyRIT/RU/KZ\n(48)",
              "Padded\n~320 tok\n(12)", "Long ~920 tok,\nattack at\nend (6)"]
    end_rows = [r for r in lc["rows"] if r["id"].endswith("-end")]
    g0 = [100 * bs[s]["G0"] / bs[s]["n"] for s in ("M72", "TTH", "OBF", "PAD")] + [100.0 * sum(r["model_grant"] for r in end_rows) / len(end_rows)]
    g1 = [100 * bs[s]["G1"] / bs[s]["n"] for s in ("M72", "TTH", "OBF", "PAD")] + [100.0 * (lc["n_end"] - lc["end_caught_truncated"]) / lc["n_end"]]
    g2 = [100 * bs[s]["G2"] / bs[s]["n"] for s in ("M72", "TTH", "OBF", "PAD")] + [100.0 * (lc["n_end"] - lc["end_caught_windowed"]) / lc["n_end"]]
    grouped(a, groups, [g0, g1, g2], [S1, S2, S3], ["G0 baseline (LLM decides)", "G1 + detector, first 512 tokens",
                                                   "G2 + detector, sliding window"], 118)
    a.set_ylabel("Attacks granted access, %")
    a.set_title("a) Unauthorized grants by attack suite", loc="left", fontsize=7.6, color=INK)
    a.legend(loc="upper left", fontsize=5.6, ncol=1)
    a.tick_params(axis="x", labelsize=5.4)

    cf = G["configs"]
    pts = [("G0 baseline", cf["G0 Baseline (LLM decides)"]), ("G2 + detector", cf["G2 + detector, windowed"]),
           ("G3 backend check", cf["G3 Backend check (Midterm)"]), ("G4 backend + detector", cf["G4 Backend + detector"])]
    style = {"G0 baseline": (S1, (-4, 7), "right", True), "G2 + detector": (S3, (6, 6), "left", True),
             "G3 backend check": (S2, (6, -12), "left", True), "G4 backend + detector": (INK2, (6, -14), "left", False)}
    for name, v in pts:
        col, off, ha, filled = style[name]
        b.scatter(v["hijack_asr"], v["legit_success"], s=40 if filled else 70, color=col if filled else "none",
                  edgecolor=col if not filled else "white", linewidth=1.0, zorder=3 if filled else 4)
        b.annotate(f"{name}: ASR {v['hijack_asr']}%, legit {v['legit_success']}%", (v["hijack_asr"], v["legit_success"]),
                   xytext=off, textcoords="offset points", fontsize=5.4, color=INK2, ha=ha)
    b.set_xlim(-5, 55)
    b.set_ylim(0, 112)
    b.set_xlabel("Hijacking success, % of 292 attacks")
    b.set_ylabel("Legitimate requests granted, %")
    b.set_title("b) Security vs. availability trade-off", loc="left", fontsize=7.6, color=INK)
    fig.tight_layout(pad=0.35, w_pad=1.2)
    fig.savefig(out / "fig1_gatekeeper.png")

    man = m["rag"]["manual"]
    order = ["R0", "R1", "R2", "R3", "R1V", "R2V", "R4"]
    lab = {"R0": "R0 baseline\nprompt", "R1": "R1 Midterm\nprotected prompt", "R2": "R2 R1 + source\nvalidation",
           "R3": "R3 R2 + injection\ndetector", "R1V": "R1V R1 + output\nverification", "R2V": "R2V R2 + output\nverification*",
           "R4": "R4 R3 + output\nverification"}
    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    grouped(ax, [lab[k] for k in order], [[man[k]["harmful_rate"] for k in order], [man[k]["task_success"] for k in order],
                                          [man[k]["hallucination_rate"] for k in order]],
            [S1, S2, S3], ["Harmful answers to 24 attacks", "Correct answers to 22 legitimate questions",
                           "Hallucinated answers to 22 legitimate questions"], 122)
    ax.set_ylabel("%")
    ax.legend(loc="upper center", fontsize=5.6, ncol=3, bbox_to_anchor=(0.5, 1.16))
    ax.tick_params(axis="x", labelsize=5.6)
    fig.tight_layout(pad=0.35)
    fig.savefig(out / "fig2_rag.png")
    print("figures written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
