"""
make_figures.py
===============
Manuscript Figures 3, 4, and 5 (internally C2, C3, C4), each with the section-5
realization spread drawn as a band. Reads the per-sweep result JSONs written by
run_sweep.py --mode finalize, plus the C4 cell store for the per-realization
predicted-transition scatter.

File-to-manuscript numbering (the filenames predate the manuscript):
    fig2_c2_range.png    -> Figure 3, range width
    fig3_c3_density.png  -> Figure 4, sample density
    fig4_c4_position.png -> Figure 5, across the Hopf

Figures are rendered at the width they occupy on the JURPA page (3.2 in) and
must be inserted at 100%. Interpretation lives in the captions and body text,
not in the plots.

    python make_figures.py
"""
import sys, os, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import sweep

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
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
    """Manuscript Figure 3 -- extrapolation distance vs training-range width."""
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
    """Manuscript Figure 4 -- extrapolation distance vs sample density."""
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
    # apart horizontally rather than staggering -- all six share one baseline.
    nudge = {0: -0.35, 1: 0.35}
    for i, (xi, pl) in enumerate(zip(x, psl)):
        ax.text(xi + nudge.get(i, 0.0), 0.02, f"{pl//1000}k",
                fontsize=5.5, ha="center", color="0.4")

    ax.legend(loc="upper right", **LEG)
    _save(fig, "fig3_c3_density")


def fig4():
    """Manuscript Figure 5 -- where the collapse is predicted, vs window position.

    Single panel. The strict across-Hopf depth is zero at every d, which the
    Results text states in one sentence; it does not need its own axes.
    """
    r = _load("c4")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])                       # d above Hopf
    # per-realization predicted-transition rho from the cell store
    trans_per, t_med, t_q1, t_q3 = [], [], [], []
    for p in pts:
        cells = [pickle.load(open(q, "rb"))
                 for q in sorted(glob.glob(os.path.join(DATA, "C4_cells", f"{p['id']}_r*.pkl")))]
        tr = [sweep.predicted_transition_rho(c) for c in cells]
        tr = [t for t in tr if t is not None]
        trans_per.append(tr)
        t_med.append(np.median(tr)); t_q1.append(np.percentile(tr, 25)); t_q3.append(np.percentile(tr, 75))
    t_med = np.array(t_med); t_q1 = np.array(t_q1); t_q3 = np.array(t_q3)

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    rng = np.random.default_rng(7)
    for xi, tr in zip(x, trans_per):
        jx = xi + rng.uniform(-0.12, 0.12, size=len(tr))
        ax.plot(jx, tr, ".", color=RED, ms=3, alpha=0.25, zorder=1)
    ax.fill_between(x, t_q1, t_q3, color=RED, alpha=0.18, zorder=2,
                    label="bootstrap IQR")
    ax.plot(x, t_med, "o-", color=RED, lw=1.4, ms=4, zorder=3,
            label="predicted collapse")
    ax.axhline(HOPF, color=DARK, ls="--", lw=1.1, label="Hopf, 24.74")
    ax.axhline(COEX, color=TEAL, ls="-.", lw=1.1, label="coexistence, 24.06")
    ax.set_xlabel(r"distance of lower edge above Hopf $d$")
    ax.set_ylabel(r"predicted collapse $\rho$")
    ax.set_xticks(x); ax.set_ylim(23.8, 25.2); ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0, **LEG)
    _save(fig, "fig4_c4_position")


if __name__ == "__main__":
    fig2(); fig3(); fig4()
