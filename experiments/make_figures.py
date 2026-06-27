"""
make_figures.py
===============
Figures 2-4 for contributions C2-C4 (methodology section 4), each with the
section-5 realization spread drawn as a band. Reads the per-sweep result JSONs
written by run_sweep.py --mode finalize, plus the C4 cell store for the
per-realization predicted-transition scatter.

    python make_figures.py
"""
import sys, os, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sweep

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
GREY, RED, TEAL, DARK = "0.55", "#c0392b", "#16a085", "#2c3e50"
HOPF, COEX = 24.74, 24.06


def _load(name):
    with open(os.path.join(FIGS, f"{name}_result.json")) as f:
        return json.load(f)


def _scatter(ax, xs, per_lists, color, jitter):
    rng = np.random.default_rng(7)
    for x, vals in zip(xs, per_lists):
        jx = x + rng.uniform(-jitter, jitter, size=len(vals))
        ax.plot(jx, vals, ".", color=color, ms=3, alpha=0.22, zorder=1)


def fig2():
    r = _load("c2")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    ceil = np.array([p["diagnostics"]["ceiling_rho"] for p in pts])
    per = [p["per_realization"] for p in pts]

    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter(ax, x, per, RED, 0.18)
    ax.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2,
                    label="bootstrap IQR (realization spread)")
    ax.plot(x, med, "o-", color=RED, lw=1.8, ms=6, zorder=3,
            label=r"median-curve $\Delta\rho_\uparrow$")
    # mark the sub-Hopf-contaminated point
    ax.annotate("W=10, lower edge ρ=24\n(dips below Hopf)",
                xy=(10, med[-1]), xytext=(7.0, med[-1] + 0.45), fontsize=7.5,
                color=DARK, arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
    ax.set_xlabel("training-range width  $W$  (ρ units, density fixed at 1/unit)")
    ax.set_ylabel(r"upward same-class extrapolation  $\Delta\rho_\uparrow$")
    ax.set_title("C2, extrapolation distance vs training-range width\n"
                 "(center ρ=29, $L_\\mathrm{total}$=120k fixed, R=32)", fontsize=10)
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    # twin axis: absolute extrapolation ceiling
    ax2 = ax.twinx()
    ax2.plot(x, ceil, "s--", color=TEAL, lw=1.2, ms=5, alpha=0.9,
             label="absolute ceiling  edge+$\\Delta\\rho_\\uparrow$")
    ax2.set_ylabel("absolute extrapolation ceiling  ρ", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_c2_range.png"), dpi=150)
    plt.close(fig); print("fig2 -> figures/fig2_c2_range.png")


def fig3():
    r = _load("c3")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    psl = np.array([p["diagnostics"]["per_sample_len"] for p in pts])
    per = [p["per_realization"] for p in pts]

    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter(ax, x, per, RED, 0.35)
    ax.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2,
                    label="bootstrap IQR (realization spread)")
    ax.plot(x, med, "o-", color=RED, lw=1.8, ms=6, zorder=3,
            label=r"median-curve $\Delta\rho_\uparrow$")
    ax.axhline(0.30, color=DARK, ls=":", lw=1.0, alpha=0.7)
    ax.annotate("saturates at $\\Delta\\rho_\\uparrow$≈0.3 from M=3",
                xy=(13, 0.30), xytext=(7, 0.50), fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
    ax.set_xlabel("number of training samples  $M$  (window $W$=6 fixed, ρ∈[26,32])")
    ax.set_ylabel(r"upward same-class extrapolation  $\Delta\rho_\uparrow$")
    ax.set_title("C3, extrapolation distance vs sample density\n"
                 "($L_\\mathrm{total}$=120k fixed so per-sample length absorbs M, R=32)",
                 fontsize=10)
    ax.set_xticks(x); ax.set_ylim(0, max(med) + 0.4); ax.grid(alpha=0.25)
    # annotate per-sample length to show the washout floor is not reached
    for xi, pl in zip(x, psl):
        ax.text(xi, 0.02, f"{pl//1000}k", fontsize=6.5, ha="center",
                color="0.4", rotation=0)
    ax.text(x.mean(), 0.085, "per-sample length in steps, washout=1000, floor not hit",
            fontsize=6.8, ha="center", color="0.4")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_c3_density.png"), dpi=150)
    plt.close(fig); print("fig3 -> figures/fig3_c3_density.png")


def fig4():
    r = _load("c4")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])                       # d above Hopf
    depth = np.array([p["median"] for p in pts])
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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: strict across-Hopf depth (the primary C4 kernel) -- zero everywhere
    axL.fill_between(x, [0]*len(x), [0]*len(x), color=RED, alpha=0.18)
    axL.plot(x, depth, "o-", color=RED, lw=1.8, ms=7,
             label="median across-Hopf depth")
    axL.axhline(0, color=DARK, ls=":", lw=1.0)
    axL.set_xlabel("distance of lower edge above Hopf  $d$")
    axL.set_ylabel("across-Hopf depth below ρ=24.74")
    axL.set_title("C4 primary kernel, strict class-match across the Hopf", fontsize=9.5)
    axL.set_xticks(x); axL.set_ylim(-0.25, 1.0); axL.grid(alpha=0.25)
    axL.text(x.mean(), 0.45, "no window crosses the Hopf cleanly\n(depth = 0 for every d)",
             fontsize=8.5, ha="center", color=DARK)
    axL.legend(loc="upper right", fontsize=8)

    # right: where the network actually places the collapse -- invariant in d
    rng = np.random.default_rng(7)
    for xi, tr in zip(x, trans_per):
        jx = xi + rng.uniform(-0.12, 0.12, size=len(tr))
        axR.plot(jx, tr, ".", color=RED, ms=3, alpha=0.25, zorder=1)
    axR.fill_between(x, t_q1, t_q3, color=RED, alpha=0.18, zorder=2,
                     label="IQR of predicted transition")
    axR.plot(x, t_med, "o-", color=RED, lw=1.8, ms=7, zorder=3,
             label="predicted collapse ρ")
    axR.axhline(HOPF, color=DARK, ls="--", lw=1.2, label="true Hopf ρ=24.74")
    axR.axhline(COEX, color=TEAL, ls="-.", lw=1.2, label="coexistence onset ρ≈24.06")
    axR.set_xlabel("distance of lower edge above Hopf  $d$")
    axR.set_ylabel("predicted collapse ρ")
    axR.set_title("C4, predicted transition overshoots the Hopf by ≈0.6,\n"
                  "independent of window position", fontsize=9.5)
    axR.set_xticks(x); axR.set_ylim(23.8, 25.2); axR.grid(alpha=0.25)
    axR.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_c4_position.png"), dpi=150)
    plt.close(fig); print("fig4 -> figures/fig4_c4_position.png")


if __name__ == "__main__":
    fig2(); fig3(); fig4()
