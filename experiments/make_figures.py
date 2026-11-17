"""
Manuscript Figures 3(a), 3(b), and 4 (internally C2, C3, C4) 
each with the realization spread drawn as a band. 
Reads the per-sweep result JSONs written by run_sweep.py --mode
finalize; c4_result.json carries the per-realization collapse points for Figure 4.

File-to-manuscript numbering (the filenames predate the manuscript):
    fig2_c2_range.png    -> Figure 3(a) (via make_merged_figure.py), range width
    fig3_c3_density.png  -> Figure 3(b) (via make_merged_figure.py), sample density
    fig4_c4_position.png -> Figure 4, across the Hopf

Figures are rendered at the width they occupy on the JURPA page (3.2 in) and
must be inserted at 100%. Interpretation lives in the captions and body text,
not in the plots.

    python make_figures.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Sized for a 3.2 in printed width at 100% insertion, so authored point sizes
# land 1:1 on the page against ~10 pt body type.
plt.rcParams.update({
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  6.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figures")
GREY, RED, TEAL, DARK = "0.55", "#c0392b", "#16a085", "#2c3e50"
HOPF, COEX = 24.74, 24.06
LEG = dict(framealpha=0.92, handlelength=1.4, borderpad=0.3, labelspacing=0.3)


def _load(name):
    with open(os.path.join(FIGS, f"{name}_result.json")) as f:
        return json.load(f)


def _save(fig, stem):
    """PNG for the repo, vector PDF for submission (JURPA accepts both).

    A file open in a viewer is locked on Windows; warn and carry on rather than
    aborting the run, but say plainly which output is now stale.
    """
    fig.tight_layout()
    written, stale = [], []
    for ext in ("png", "pdf"):
        try:
            fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=300)
            written.append("." + ext)
        except PermissionError:
            stale.append("." + ext)
    plt.close(fig)
    print(f"{stem} -> {', '.join(written) if written else '(nothing written)'}")
    if stale:
        print(f"  !! {stem}{'/'.join(stale)} LOCKED (open in a viewer?) "
              f"-- LEFT STALE, close it and re-run")


def _scatter(ax, xs, per_lists, color, jitter):
    rng = np.random.default_rng(7)
    for x, vals in zip(xs, per_lists):
        jx = x + rng.uniform(-jitter, jitter, size=len(vals))
        ax.plot(jx, vals, ".", color=color, ms=3, alpha=0.22, zorder=1)


def fig2():
    """Manuscript Figure 3(a): extrapolation distance vs training-range width."""
    r = _load("c2")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    ceil = np.array([p["diagnostics"]["ceiling_rho"] for p in pts])
    per = [p["per_realization"] for p in pts]

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    _scatter(ax, x, per, RED, 0.18)
    ax.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2,
                    label="bootstrap IQR")
    ax.plot(x, med, "o-", color=RED, lw=1.4, ms=4, zorder=3,
            label=r"median $\Delta\rho_\uparrow$")
    ax.set_xlabel(r"training-range width $W$ ($\rho$ units)")
    ax.set_ylabel(r"same-class $\Delta\rho_\uparrow$")
    ax.set_xticks(x)
    # headroom for the legend; the sub-Hopf W=10 point is discussed in the text
    top = max(max(v) for v in per)
    ax.set_ylim(0, max(top, med.max()) * 1.25)
    ax.grid(alpha=0.25)

    # twin axis: absolute extrapolation ceiling
    ax2 = ax.twinx()
    ax2.plot(x, ceil, "s--", color=TEAL, lw=1.1, ms=3.5, alpha=0.9,
             label="absolute ceiling")
    ax2.set_ylabel(r"absolute ceiling $\rho$", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", **LEG)
    _save(fig, "fig2_c2_range")


def fig3():
    """Manuscript Figure 3(b): extrapolation distance vs sample density."""
    r = _load("c3")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    psl = np.array([p["diagnostics"]["per_sample_len"] for p in pts])
    per = [p["per_realization"] for p in pts]

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    _scatter(ax, x, per, RED, 0.35)
    ax.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2,
                    label="bootstrap IQR")
    ax.plot(x, med, "o-", color=RED, lw=1.4, ms=4, zorder=3,
            label=r"median $\Delta\rho_\uparrow$")
    ax.axhline(0.30, color=DARK, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"number of training samples $M$")
    ax.set_ylabel(r"same-class $\Delta\rho_\uparrow$")
    ax.set_xticks(x); ax.set_ylim(0, max(med) + 0.4); ax.grid(alpha=0.25)

    # per-sample segment length, in steps; the caption explains what these are.
    # M=2 and M=3 sit close enough on a 3.2 in axis to touch, so nudge that pair
    # apart horizontally rather than staggering; all six share one baseline.
    nudge = {0: -0.35, 1: 0.35}
    for i, (xi, pl) in enumerate(zip(x, psl)):
        ax.text(xi + nudge.get(i, 0.0), 0.02, f"{pl//1000}k",
                fontsize=5.5, ha="center", color="0.4")

    ax.legend(loc="upper right", **LEG)
    _save(fig, "fig3_c3_density")


def fig4():
    """Manuscript Figure 4: where each network's chaotic attractor collapses, vs window position.

    Single panel. The strict across-Hopf depth is zero at every d, which the
    Results text states in one sentence; it does not need its own axes.
    Dots are the networks that settle, with each window's median and
    25th-75th percentile drawn over them. Networks still chaotic at the floor of
    the walk have no collapse point, so they are counted in a strip below the
    floor instead of being dropped. Everything is labelled in place; there is no
    legend to crowd the landmark lines.
    """
    r = _load("c4")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])                       # d above Hopf
    diags = [p["diagnostics"] for p in pts]
    floor = diags[0]["floor"]
    trans = [d["transition_rho_per_realization"] for d in diags]
    settled = [[t for t in tr if t is not None] for tr in trans]
    never = [d["n_never_settled"] for d in diags]
    t_med = np.array([d["pred_transition_rho_median"] for d in diags])
    t_q1 = np.array([d["pred_transition_rho_q1"] for d in diags])
    t_q3 = np.array([d["pred_transition_rho_q3"] for d in diags])

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    rng = np.random.default_rng(7)
    for xi, tr in zip(x, settled):
        jx = xi + rng.uniform(-0.22, 0.22, size=len(tr))
        ax.plot(jx, tr, ".", color=RED, ms=2.6, alpha=0.4, mec="none", zorder=1)
    ax.plot(x, t_med, "-", color=RED, lw=0.9, alpha=0.7, zorder=2)
    ax.errorbar(x, t_med, yerr=[t_med - t_q1, t_q3 - t_med], fmt="o", color=RED,
                ms=4, elinewidth=1.4, capsize=2.5, capthick=1.1, zorder=3)

    left = x[0] - 0.86
    ax.set_xlim(left, x[-1] + 0.69)
    lo = floor - 1.1
    ax.set_ylim(lo, max([HOPF] + [max(s) for s in settled if s]) + 1.3)

    # landmark lines, labelled at the left where the data sit lowest
    ax.axhline(HOPF, color=DARK, ls="--", lw=0.9, zorder=0)
    ax.axhline(COEX, color=TEAL, ls="-.", lw=0.9, zorder=0)
    ax.text(left + 0.15, HOPF + 0.15, f"Hopf {HOPF}", fontsize=6.5,
            color=DARK, va="bottom")
    ax.text(left + 0.15, COEX - 0.15, f"coexistence onset {COEX}", fontsize=6.5,
            color=TEAL, va="top")

    # networks still chaotic at the floor, counted in an overflow strip below it
    mid = (lo + floor) / 2
    ax.axhspan(lo, floor, color="0.93", zorder=0)
    for xi, n, tr in zip(x, never, trans):
        ax.text(xi, mid, f"{n}/{len(tr)}", ha="center", va="center",
                fontsize=6.3, color="0.3")
    axis_side = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.025, mid, f"<{floor:g}", transform=axis_side, ha="right",
            va="center", fontsize=7)

    ax.set_yticks([t for t in ax.get_yticks() if floor < t < HOPF])
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel(r"distance of lower edge above Hopf $d$")
    ax.set_ylabel(r"predicted collapse $\rho$")
    _save(fig, "fig4_c4_position")


if __name__ == "__main__":
    fig2(); fig3(); fig4()
