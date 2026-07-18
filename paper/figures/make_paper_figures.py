"""
Honest figure generation for the DP-SGD tightness-gap paper.

Reads results directly from
    codex/results/framework_validation/framework_validation_summary.json
and produces three figures into the same directory as this script:
    fig1_accountant_comparison.png
    fig2_theoretical_vs_empirical.png
    fig3_tightness_summary.png   (the "final figure" — replaces the
                                   single-bar tightness chart in the PDF)

Design goals (vs. the previous figures):
  * Plot BOTH the Wilson-supported conservative ε_lower AND the optimistic
    point estimate, with a clear visual distinction. The previous figures
    showed only one number and visually overstated the headline result.
  * Add Wilson CI-style error bars on the conservative bar so the reader
    can see uncertainty rather than a single deterministic value.
  * Annotate every bar with sample budget and the framework's own
    `result_trust` tag (exploratory / provisional / trusted) so a reader
    can immediately tell which numbers are claims and which are pilots.
  * Label x-ticks with two lines (dataset on top, auditor on bottom) so
    nothing has to be rotated; readable at small width.
  * Cap the y-axis at the PLD upper bound + a small margin and draw a
    dashed reference line at the upper bound, so 'tightness' is visually
    obvious without an extra figure.

Run:
    python paper/figures/make_paper_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]                       # .../EECE_608
RESULTS_PATH = (
    REPO_ROOT
    / "codex"
    / "results"
    / "framework_validation"
    / "framework_validation_summary.json"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_audit_rows() -> list[dict]:
    """Pull the per-(dataset, auditor) audit rows out of the validation summary."""
    payload = json.loads(RESULTS_PATH.read_text())

    # The summary file is a dict; the per-row records live under one of a
    # few known keys depending on which version produced it. Try the
    # obvious ones, then fall back to a recursive search.
    for key in ("attack_rows", "audit_rows", "rows", "results"):
        if isinstance(payload, dict) and key in payload and isinstance(payload[key], list):
            return [r for r in payload[key] if isinstance(r, dict)]

    # Fallback: walk the structure and collect the first list of dicts that
    # looks like audit rows.
    def walk(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "epsilon_upper_tighter" in obj[0]:
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                hit = walk(v)
                if hit is not None:
                    return hit
        if isinstance(obj, list):
            for v in obj:
                hit = walk(v)
                if hit is not None:
                    return hit
        return None

    rows = walk(payload)
    if rows is None:
        raise RuntimeError(f"Could not locate audit rows inside {RESULTS_PATH}")
    return rows


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Symmetric half-width of a Wilson CI for plotting error bars only."""
    if n <= 0 or not (0.0 <= p <= 1.0):
        return 0.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    # Convert TPR/FPR uncertainty into an approximate ε uncertainty bar.
    # We just use 'margin' on the probability scale as a rough indicator;
    # for a rigorous bar, recompute log((TPR-margin)/(FPR+margin)).
    return margin


def epsilon_lower_ci_halfwidth(row: dict) -> float:
    """Approximate ε-scale half-width from TPR/FPR Wilson CIs.

    This is a *visual* uncertainty indicator — the framework's actual
    epsilon_lower_conservative is the validated number; this bar just
    shows how much the value would move if TPR went down or FPR went up
    by a Wilson margin.
    """
    tpr = row.get("selected_tpr")
    fpr = row.get("selected_fpr")
    n_m = row.get("num_member_samples") or 0
    n_n = row.get("num_nonmember_samples") or 0
    if tpr is None or fpr is None:
        return 0.0

    dtpr = wilson_halfwidth(tpr, n_m)
    dfpr = wilson_halfwidth(fpr, n_n)

    tpr_lo = max(1e-6, tpr - dtpr)
    fpr_hi = min(1.0 - 1e-6, fpr + dfpr)
    if fpr_hi <= 0 or tpr_lo <= 0:
        return 0.0
    eps_pess = max(0.0, math.log(tpr_lo / fpr_hi))
    eps_point = max(0.0, math.log(max(tpr, 1e-6) / max(fpr, 1e-6)))
    return max(0.0, eps_point - eps_pess)


# ---------------------------------------------------------------------------
# Figure 1 — accountant comparison
# ---------------------------------------------------------------------------

def figure_accountant_comparison(rows: list[dict], out_path: Path) -> None:
    """RDP vs PLD upper bounds, one cluster per dataset.

    We use only the training-row epsilon_upper values so the comparison
    isn't entangled with audit-specific runs.
    """
    by_dataset: dict[str, dict[str, float]] = {}
    for r in rows:
        ds = r.get("dataset")
        rdp = r.get("epsilon_upper_rdp")
        pld = r.get("epsilon_upper_tighter")
        if ds is None or rdp is None or pld is None:
            continue
        # Keep the first occurrence per dataset; all rows for a given
        # dataset share the same training-side ε_upper in this project.
        by_dataset.setdefault(ds, {"rdp": rdp, "pld": pld})

    datasets = sorted(by_dataset.keys())
    rdp_vals = [by_dataset[d]["rdp"] for d in datasets]
    pld_vals = [by_dataset[d]["pld"] for d in datasets]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(datasets))
    width = 0.36

    b1 = ax.bar(x - width / 2, rdp_vals, width, label="RDP", color="#4C78A8")
    b2 = ax.bar(x + width / 2, pld_vals, width, label="PLD (used as reference)", color="#F58518")

    for bar, val in zip(b1, rdp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, pld_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"theoretical upper bound  $\varepsilon_{\mathrm{upper}}$")
    ax.set_title("Accountant comparison: RDP vs PLD upper bounds")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(rdp_vals), max(pld_vals)) * 1.18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — theoretical vs empirical (point + Wilson)
# ---------------------------------------------------------------------------

TRUST_COLOR = {
    "trusted": "#1B7F3A",
    "provisional": "#F58518",
    "exploratory": "#B5B5B5",
}


def short_label(row: dict) -> tuple[str, str]:
    """Return (top_line, bottom_line) for a two-line x-tick label."""
    ds = row.get("dataset", "?")
    name = row.get("attack_name", "?")
    pretty = {
        "passive_negative_loss": "passive\nneg-loss",
        "passive_negative_loss_matched": "passive\nneg-loss (matched)",
        "passive_raw_lira": "passive\nraw LiRA",
        "canary_random": "canary\nrandom",
    }.get(name, name.replace("_", "\n"))
    return ds, pretty


def figure_theoretical_vs_empirical(rows: list[dict], out_path: Path) -> None:
    """For each (dataset, auditor) row, plot the PLD upper, the Wilson
    conservative lower (with a CI bar), and the optimistic point estimate
    as a hatched overlay. Sample size and trust tag annotated under each
    cluster.
    """
    # Keep only the rows we'll actually plot (skip 'not_supported' rows).
    rows = [r for r in rows if r.get("status") == "ok"]
    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("attack_name", "")))

    n = len(rows)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * n), 5.2))
    x = np.arange(n)
    w_upper = 0.30
    w_lower = 0.30

    # PLD upper bound (left bar in each cluster)
    upper_vals = [r.get("epsilon_upper_tighter", 0.0) or 0.0 for r in rows]
    ax.bar(x - w_lower / 2, upper_vals, w_lower, color="#4C78A8",
           label=r"PLD upper $\varepsilon_{\mathrm{upper}}$")

    # Wilson conservative lower (right bar) with error bars
    cons_vals = [r.get("epsilon_lower_conservative", 0.0) or 0.0 for r in rows]
    cons_err = [epsilon_lower_ci_halfwidth(r) for r in rows]
    bar_colors = [TRUST_COLOR.get(r.get("result_trust", "exploratory"), "#B5B5B5") for r in rows]
    ax.bar(x + w_upper / 2, cons_vals, w_upper, color=bar_colors,
           yerr=cons_err, capsize=3, ecolor="black",
           label=r"Wilson-supported $\varepsilon_{\mathrm{lower}}$")

    # Point estimate as a hatched outline overlay on top of the Wilson bar.
    # Plotting a transparent bar with hatch makes the point estimate
    # visible without competing visually with the validated value.
    point_vals = [r.get("epsilon_lower_point", 0.0) or 0.0 for r in rows]
    ax.bar(x + w_upper / 2, point_vals, w_upper,
           facecolor="none", edgecolor="black", linewidth=0.8,
           hatch="///", label="point estimate (not validated)")

    # X-tick labels: dataset on top line, auditor on bottom line.
    ds_line = []
    auditor_line = []
    for r in rows:
        ds, pretty = short_label(r)
        ds_line.append(ds)
        auditor_line.append(pretty)
    ax.set_xticks(x)
    ax.set_xticklabels(auditor_line, fontsize=8)
    # Add the dataset name as a second-row annotation under each tick.
    for xi, ds in zip(x, ds_line):
        ax.annotate(ds, xy=(xi, 0), xytext=(0, -34),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8, color="#444444")

    # Sample-size + trust annotation under each cluster
    for xi, r in zip(x, rows):
        n_m = r.get("num_member_samples")
        n_n = r.get("num_nonmember_samples")
        trust = r.get("result_trust", "?")
        support = "n/a"
        if n_m is not None and n_n is not None:
            support = f"n={n_m}/{n_n}"
        ax.annotate(f"{support}\n[{trust}]", xy=(xi, 0), xytext=(0, -52),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="#666666")

    ax.set_ylabel(r"privacy loss $\varepsilon$")
    ax.set_title("Theoretical PLD upper bound vs validated empirical lower bound")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ymax = max(max(upper_vals + point_vals + cons_vals) * 1.15, 0.5)
    ax.set_ylim(0, ymax)

    # Make space at the bottom for the two-line x labels + annotations.
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — tightness summary  (THE final figure)
# ---------------------------------------------------------------------------

def figure_tightness_summary(rows: list[dict], out_path: Path) -> None:
    """Tightness ratio per (dataset, auditor), plotted honestly.

    Fixes the original figure 3 problems:
      * Original: a single bar at ~100% with all others at 0%, no CIs,
        no distinction between Wilson and point estimate.
      * New: paired bars per row (Wilson conservative vs point estimate),
        100% reference line, sample-size + trust annotations, color-coded
        by trust.
    """
    rows = [r for r in rows if r.get("status") == "ok"
            and r.get("epsilon_upper_tighter")]
    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("attack_name", "")))

    if not rows:
        return

    cons_ratio = []
    point_ratio = []
    labels_top = []
    labels_bot = []
    annotations = []
    bar_colors = []

    for r in rows:
        upper = r["epsilon_upper_tighter"]
        cons = (r.get("epsilon_lower_conservative") or 0.0) / upper
        point = (r.get("epsilon_lower_point") or 0.0) / upper
        cons_ratio.append(min(cons, 1.4))   # clip pathological >1 to a visual ceiling
        point_ratio.append(min(point, 1.4))

        ds, pretty = short_label(r)
        labels_top.append(ds)
        labels_bot.append(pretty)

        n_m = r.get("num_member_samples") or 0
        n_n = r.get("num_nonmember_samples") or 0
        trust = r.get("result_trust", "?")
        annotations.append(f"n={n_m}/{n_n}  [{trust}]")
        bar_colors.append(TRUST_COLOR.get(trust, "#B5B5B5"))

    n = len(rows)
    x = np.arange(n)
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(8.5, 1.3 * n), 5.0))

    bars_cons = ax.bar(
        x - w / 2, [v * 100 for v in cons_ratio], w,
        color=bar_colors, label="Wilson-supported (validated)",
        edgecolor="black", linewidth=0.4,
    )
    ax.bar(
        x + w / 2, [v * 100 for v in point_ratio], w,
        facecolor="none", edgecolor="black", linewidth=0.8,
        hatch="///", label="point estimate (not validated)",
    )

    # 100% reference line
    ax.axhline(100, color="#222222", linestyle="--", linewidth=1.0)
    ax.text(n - 0.5, 101.5, "100% = saturates PLD bound",
            ha="right", va="bottom", fontsize=8, color="#222222")

    # Numeric labels on the validated bars
    for bar, v in zip(bars_cons, cons_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bot, fontsize=8)
    for xi, ds in zip(x, labels_top):
        ax.annotate(ds, xy=(xi, 0), xytext=(0, -34),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8, color="#444444")
    for xi, ann in zip(x, annotations):
        ax.annotate(ann, xy=(xi, 0), xytext=(0, -52),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="#666666")

    ax.set_ylabel(r"tightness ratio $\rho = \varepsilon_{\mathrm{lower}} / \varepsilon_{\mathrm{upper}}$  (%)")
    ax.set_title("How much of the PLD upper bound each auditor recovers")
    ax.set_ylim(0, max(140, max(point_ratio) * 100 * 1.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend that also explains the trust colors
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles += [
        Patch(color=TRUST_COLOR["trusted"], label="trust: trusted"),
        Patch(color=TRUST_COLOR["provisional"], label="trust: provisional"),
        Patch(color=TRUST_COLOR["exploratory"], label="trust: exploratory"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              fontsize=8, ncol=2)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    rows = load_audit_rows()
    print(f"Loaded {len(rows)} audit rows from {RESULTS_PATH}")

    fig1 = HERE / "fig1_accountant_comparison.png"
    fig2 = HERE / "fig2_theoretical_vs_empirical.png"
    fig3 = HERE / "fig3_tightness_summary.png"

    figure_accountant_comparison(rows, fig1)
    figure_theoretical_vs_empirical(rows, fig2)
    figure_tightness_summary(rows, fig3)

    print(f"Wrote: {fig1}")
    print(f"Wrote: {fig2}")
    print(f"Wrote: {fig3}")


if __name__ == "__main__":
    main()
