"""
sweep.py
========
C2-C4 sweep machinery: the extrapolation-distance measurement of methodology
section 3.5 and the three sampling sweeps of section 4, run against the locked
architecture (ESNConfig defaults = post-gate locked values, methodology v2 1.5).

The C1 gate is closed (Session 6); this module never touches a hyperparameter.
It varies only the three sampling choices the literature fixes for convenience:

    C2  training-range WIDTH      (density held at 1 sample / rho-unit)
    C3  sample DENSITY            (window width held at W = 6)
    C4  window POSITION           (width W = 4, count M = 5 held)

Every configuration holds the total training length L_total = 120,000 ESN steps,
so when the sample count M changes the per-sample length L_total // M absorbs it
and the x-axis is sample count, not data volume (methodology 4, C3).

Extrapolation distance Delta-rho (methodology 3.5)
--------------------------------------------------
Starting at a training-window edge, step outward in rho at resolution
delta-rho = 0.1 and find the largest contiguous outward run for which a per-rho
validity test holds. Validity has two forms, chosen by what the test rho crosses:

  same-class (C2/C3, staying in the chaotic band, stepping UP into higher-rho
      chaos): valid if the valid-prediction time stays at or above a fixed
      fraction FRAC of the in-window VPT and the predicted class is still
      chaotic and ground truth is still chaotic. VPT is read in raw time units;
      the ratio is unit-free, so the costly per-rho Lyapunov-time conversion is
      skipped here.

  across-bifurcation (C4, stepping DOWN across the Hopf at rho ~ 24.74 into the
      fixed-point regime): VPT is not meaningful once the attractor type changes,
      so validity is the qualitative-class match alone (methodology 3.3, 3.5).
      Ground-truth class below the Hopf is taken from the section-2.3 landmark
      (fixed_point for rho < 24.74), not the single-IC classifier, because the
      coexistence sliver fools a single-IC label -- the same reasoning the v2
      C1 revision used.

Aggregation (methodology 3.5, 5)
--------------------------------
Validity is evaluated both per realization (giving an R-sample Delta-rho
distribution -> median + IQR band, the section-5 figure quantity) and on the
median-over-realizations curve (the section-3.5 point estimate). The two are
reported side by side; agreement is the internal consistency check.
"""

from __future__ import annotations
import numpy as np

import lorenz
import metrics
from reservoir import _normalize_param

# ---- measurement constants (pre-registered, logged in the progress log) ---- #
DELTA_RHO = 0.1            # outward step resolution (methodology 4)
FRAC = 0.5                 # same-class VPT must stay >= FRAC * in-window VPT
HOPF_RHO = 24.74           # subcritical Hopf landmark (methodology 2.3)
VPT_WARM = 200             # teacher-forced warmup steps for warmup-then-free-run
VPT_FREE = 1500            # free-run steps for the VPT comparison
CLASS_FREE = 3000          # free-run steps for a cold-extrapolation class call
CLASS_DISCARD = 1000       # opening transient discarded from a cold run
BREAK_RUN = 6              # stop a walk after this many consecutive invalid steps
DRHO_UP_MAX = 6.0          # cap on the upward same-class walk (rho units)
RHO_FLOOR = 20.0           # downward walk floor for the across-Hopf test


# --------------------------------------------------------------------------- #
# sweep definitions (methodology section 4)                                   #
# --------------------------------------------------------------------------- #
def _linspace_rhos(center, width, M):
    return list(np.round(np.linspace(center - width / 2.0, center + width / 2.0, M), 4))


def sweep_points(which: str):
    """
    Return the list of base configurations for a sweep, each a dict carrying the
    independent-variable value and the training rho set. L_total is held fixed
    elsewhere; per-sample length is L_total // M.
    """
    if which == "C2":
        # range width: density fixed at 1 sample / rho-unit -> M = W + 1
        center = 29.0
        out = []
        for W in (2, 4, 6, 8, 10):
            M = W + 1
            out.append(dict(id=f"C2_W{W}", x=float(W), xlabel="W",
                            center=center, width=float(W), M=M,
                            train_rhos=_linspace_rhos(center, W, M),
                            direction="up"))
        return out
    if which == "C2clamp":
        # Robustness check for the W=10 / section-2.4 tension (Session 8).
        # C2_W10's window [24, 34] puts its lowest training sample at rho = 24,
        # below the Hopf at 24.74, which section 2.4 excludes from training. This
        # variant is that same window -- same center, width, upper edge, and
        # upward-walk geometry -- but with the training samples clamped to
        # rho >= HOPF_RHO, which drops the single sub-Hopf sample at rho = 24.
        # Comparing its Delta-rho to C2_W10's tells us whether the W=10 jump is
        # contamination from that sample or a real feature of a wide window.
        center, W = 29.0, 10.0
        base = _linspace_rhos(center, W, int(W) + 1)      # [24, 25, ..., 34]
        clamped = [r for r in base if r >= HOPF_RHO]      # [25, ..., 34], M = 10
        return [dict(id="C2_W10c", x=float(W), xlabel="W",
                     center=center, width=W, M=len(clamped),
                     train_rhos=clamped, direction="up")]
    if which == "C3":
        # density: window width fixed at W = 6 centred on 29 (rho in [26, 32])
        center, W = 29.0, 6.0
        out = []
        for M in (2, 3, 5, 8, 13, 21):
            out.append(dict(id=f"C3_M{M}", x=float(M), xlabel="M",
                            center=center, width=W, M=M,
                            train_rhos=_linspace_rhos(center, W, M),
                            direction="up"))
        return out
    if which == "C4":
        # position: width W = 4, count M = 5; slide the centre across the Hopf
        W, M = 4.0, 5
        out = []
        for center in (27.0, 29.0, 31.0, 33.0):
            lower_edge = center - W / 2.0
            d = round(lower_edge - HOPF_RHO, 4)   # distance of lower edge above Hopf
            out.append(dict(id=f"C4_c{int(center)}", x=float(d), xlabel="d",
                            center=center, width=W, M=M,
                            train_rhos=_linspace_rhos(center, W, M),
                            lower_edge=lower_edge, direction="down"))
        return out
    raise ValueError(which)


# --------------------------------------------------------------------------- #
# per-(config, realization) cell measurement                                  #
# --------------------------------------------------------------------------- #
def _vpt_raw(pred, truth_free, vgtt):
    """VPT in RAW time units (in_lyap_times=False), capped at the VGTT."""
    return metrics.valid_prediction_time(pred, truth_free, eps=metrics.EPS,
                                         vgtt=vgtt, in_lyap_times=False)


def _class_of(traj):
    """Qualitative class from a free-run trajectory (return-map spread, no LE)."""
    if traj is None or not np.all(np.isfinite(traj)) or np.max(np.abs(traj)) > 1e4:
        return "fixed_point"
    return lorenz.classify_trajectory(traj, lyap=None)


def measure_up(esn, spec, truth):
    """
    Same-class upward measurement for one realization (C2/C3). Walks rho upward
    from the upper window edge; at each step a single warmup-then-free-run gives
    both the VPT (vs cached ground truth) and the predicted class (from the
    free-run portion). Returns aligned arrays on a fixed rho grid plus the
    in-window reference VPT.
    """
    center = spec["center"]
    upper_edge = center + spec["width"] / 2.0

    # in-window reference VPT at the window centre
    tc = truth[round(center, 2)]
    pred_in = esn.warmup_then_freerun(tc["warm"], center, VPT_FREE)
    vpt_in = _vpt_raw(pred_in, tc["free"], tc["vgtt"])

    grid = np.round(np.arange(upper_edge + DELTA_RHO,
                              upper_edge + DRHO_UP_MAX + 1e-9, DELTA_RHO), 2)
    vpt = np.zeros(len(grid))
    pcls = np.empty(len(grid), dtype=object)
    tcls = np.empty(len(grid), dtype=object)
    consec_bad = 0
    for i, rho in enumerate(grid):
        t = truth.get(round(float(rho), 2))
        if t is None:
            pcls[i] = "fixed_point"; tcls[i] = "fixed_point"; vpt[i] = 0.0
        else:
            pred = esn.warmup_then_freerun(t["warm"], float(rho), VPT_FREE)
            vpt[i] = _vpt_raw(pred, t["free"], t["vgtt"])
            pcls[i] = _class_of(pred)            # free-run portion -> class
            tcls[i] = t["class"]
        valid = (vpt[i] >= FRAC * vpt_in) and (pcls[i] == "chaotic") and (tcls[i] == "chaotic")
        consec_bad = 0 if valid else consec_bad + 1
        if consec_bad >= BREAK_RUN:
            # past a sustained break: remaining points are invalid by definition
            for j in range(i + 1, len(grid)):
                vpt[j] = 0.0; pcls[j] = "fixed_point"; tcls[j] = truth.get(
                    round(float(grid[j]), 2), {"class": "fixed_point"})["class"]
            break
    return dict(edge=float(upper_edge), grid=grid, vpt=vpt, vpt_in=float(vpt_in),
                pred_class=pcls, true_class=tcls)


def measure_down(esn, spec, truth, primer_hat):
    """
    Across-Hopf downward measurement for one realization (C4). Walks rho downward
    from the lower window edge; at each step a cold extrapolation (no ground
    truth at the target rho) gives the predicted class, compared to the landmark
    ground-truth class. Returns aligned arrays on a fixed rho grid.
    """
    lower_edge = spec["lower_edge"]
    grid = np.round(np.arange(lower_edge - DELTA_RHO, RHO_FLOOR - 1e-9,
                              -DELTA_RHO), 2)
    pcls = np.empty(len(grid), dtype=object)
    tcls = np.empty(len(grid), dtype=object)
    consec_bad = 0
    for i, rho in enumerate(grid):
        tr = esn.cold_extrapolate(float(rho), n_free=CLASS_FREE,
                                  discard=CLASS_DISCARD, primer_hat=primer_hat)
        pcls[i] = _class_of(tr)
        # landmark ground-truth class (methodology 2.3): chaotic >= Hopf else FP
        tcls[i] = "chaotic" if rho >= HOPF_RHO else "fixed_point"
        valid = (pcls[i] == tcls[i])
        consec_bad = 0 if valid else consec_bad + 1
        if consec_bad >= BREAK_RUN:
            for j in range(i + 1, len(grid)):
                pcls[j] = "chaotic" if grid[j] >= HOPF_RHO else "fixed_point"
                # mark as mismatch by flipping the predicted label
                pcls[j] = "chaotic" if tcls[j] == "fixed_point" else "fixed_point"
                tcls[j] = "chaotic" if grid[j] >= HOPF_RHO else "fixed_point"
            break
    return dict(edge=float(lower_edge), grid=grid, pred_class=pcls, true_class=tcls)


# --------------------------------------------------------------------------- #
# Delta-rho from an aligned arrays cell                                        #
# --------------------------------------------------------------------------- #
def delta_rho_up(cell, vpt_in=None):
    """Contiguous same-class upward distance from the edge for one realization."""
    vin = cell["vpt_in"] if vpt_in is None else vpt_in
    n = 0
    for i in range(len(cell["grid"])):
        ok = (cell["vpt"][i] >= FRAC * vin) and (cell["pred_class"][i] == "chaotic") \
             and (cell["true_class"][i] == "chaotic")
        if ok:
            n += 1
        else:
            break
    return round(n * DELTA_RHO, 4)


def across_hopf_depth(cell):
    """
    Across-Hopf reach for one realization: how far BELOW the Hopf the predicted
    class still matches ground truth, contiguously from the lower edge. 0 if the
    network never reaches a correct fixed-point label past the Hopf.
    """
    grid, pcls, tcls = cell["grid"], cell["pred_class"], cell["true_class"]
    lowest_matched = None
    for i in range(len(grid)):
        if pcls[i] == tcls[i]:
            lowest_matched = grid[i]
        else:
            break
    if lowest_matched is None:
        return 0.0
    return round(max(0.0, HOPF_RHO - float(lowest_matched)), 4)


def crossed_hopf(cell):
    """True if the contiguous class-match from the edge reaches below the Hopf."""
    return across_hopf_depth(cell) > 1e-9


def predicted_transition_rho(cell):
    """
    The network's own predicted collapse location: the highest rho at/below which
    the predicted class is fixed_point for the whole remaining downward walk. This
    is where the ESN, riding the chaotic manifold downward, finally settles onto a
    fixed point. Returns None if it never settles. The overshoot relative to the
    true Hopf (HOPF_RHO - this) measures how far past the Hopf the spurious chaotic
    attractor persists (the pseudo-Lorenz persistence of methodology 9).
    """
    grid, pcls = cell["grid"], cell["pred_class"]
    trans = None
    for i in range(len(grid) - 1, -1, -1):       # scan upward from the lowest rho
        if pcls[i] == "fixed_point":
            trans = float(grid[i])
        else:
            break
    return trans
