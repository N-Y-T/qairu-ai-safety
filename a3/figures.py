"""Report figures from metrics.json (static PNG for the PDF report). Palette: validated slots 1-3 + neutrals."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 220})


def bars(ax, groups, series, colors, labels, ylim=None, fmt="{:.0f}"):
    n = len(series)
    w = 0.8 / n
    for i, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        b = ax.bar(xs, vals, w * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.6)
        for x, v in zip(xs, vals):
            if v is not None:
                ax.text(x, v + (ylim[1] if ylim else 100) * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)


def main(metrics_path, outdir):
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    G = m["gatekeeper"]
    bs = G["by_suite"]

    # Figure 1: gatekeeper hijacking success by suite and detector recall by evasion family
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1, 1.35]})
    suites = ["M72", "TTH", "OBF", "PAD"]
    names = {"M72": f"Midterm\n(n={bs['M72']['n']})", "TTH": f"Tensor Trust\n(n={bs['TTH']['n']})",
             "OBF": f"PyRIT/translated\n(n={bs['OBF']['n']})", "PAD": f"Long padding\n(n={bs['PAD']['n']})"}
    ser = [[100 * bs[s][k] / bs[s]["n"] for s in suites] for k in ("G0", "G1", "G2")]
    top = max(max(x) for x in ser)
    bars(a, [names[s] for s in suites], ser, [S1, S2, S3], ["G0 baseline", "G1 detector (truncated)", "G2 detector (windowed)"],
         ylim=(0, max(10, top * 1.25)))
    a.set_ylabel("Unauthorized grants, %")
    a.set_title("a) Hijacking success by test suite", loc="left", fontsize=8, color=INK)
    a.legend(loc="upper right", fontsize=6)
    fam = G["families"]
    order = [k for k in fam]
    short = {"manual_kazakh": "Kazakh", "manual_russian": "Russian", "padding_end": "Pad: attack\nat end",
             "padding_start": "Pad: attack\nat start", "pyrit_base64": "Base64", "pyrit_character_space": "Char\nspace",
             "pyrit_leetspeak": "Leet", "pyrit_rot13": "ROT13", "pyrit_unicode_confusable": "Unicode\nconfus.",
             "pyrit_zero_width": "Zero\nwidth"}
    bars(b, [short.get(k, k) for k in order], [[fam[k]["recall_trunc"] for k in order], [fam[k]["recall_win"] for k in order]],
         [S2, S3], ["truncated (512 tokens)", "windowed"], ylim=(0, 118))
    b.set_ylabel("Detector recall, %")
    b.set_title("b) Detector recall on evasion variants (6 seeds each)", loc="left", fontsize=8, color=INK)
    b.legend(loc="upper left", fontsize=6, ncol=2)
    b.tick_params(axis="x", labelsize=5.8)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig1_gatekeeper.png")

    # Figure 2: RAG configurations
    rc = m["rag"]["configs"]
    keys = list(rc.keys())
    lab = {"R0 Baseline prompt": "R0\nbaseline", "R1 Midterm protected prompt": "R1\nprotected\nprompt",
           "R2 + source validation": "R2\n+source\nvalidation", "R2 + injection detector": "R3\n+injection\ndetector",
           "R1 + output verification only": "R1V\nverifier\nonly", "R4 Full pipeline": "R4\nfull\npipeline"}
    fig, ax = plt.subplots(figsize=(7.1, 2.05))
    bars(ax, [lab[k] for k in keys], [[rc[k]["asr"] for k in keys], [rc[k]["task_success"] for k in keys],
                                      [rc[k]["hallucination_rate"] for k in keys]],
         [S1, S2, S3], ["Attack success rate (24 attacks)", "Legitimate task success (22 controls)",
                        "Hallucination rate, judge (18 answerable)"], ylim=(0, 118))
    ax.set_ylabel("%")
    ax.legend(loc="upper center", fontsize=6, ncol=3, bbox_to_anchor=(0.5, 1.13))
    ax.tick_params(axis="x", labelsize=6.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig2_rag.png")
    print("figures written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
