"""
diag_one.py
===========
Session 6 diagnostic. Evaluate ONE C1 configuration while varying a
*non-architecture* knob, and append the result to data/diag_log.json.

Two knobs, neither of which is a section-8 gate lever:
  (a) the parameter-channel reference interval (reservoir.RHO_REF), recentred
      and/or rescaled via --ref-lo / --ref-hi;
  (b) the cold-extrapolation primer of methodology 1.4, via --primer
      {nearest, central, edge, none} and --primer-len.

The architecture (N, spectral radius, leak, ridge, gamma_in, gamma_p) is held at
the Session-5 best-found config unless overridden, so any movement is attributable
to the knob under test, not to a hyperparameter change.

One config per process keeps each run inside the time budget (background jobs do
not persist here), exactly as the section-8 gate walk was run.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pickle
import lorenz
import reservoir
import training
from reservoir import ESNConfig

RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]
RHO_GRID = np.round(np.arange(24.0, 32.0 + 1e-9, 0.1), 2)
L_TOTAL = 120_000
MASTER = 20260613
LOG = os.path.join(os.path.dirname(__file__), "..", "data", "diag_log.json")


def load_truth():
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"ground_truth_{MASTER+777}.pkl")
    with open(cache, "rb") as f:
        return pickle.load(f)


def predicted_bifurcation_primed(esn, rho_grid, segments, primer_mode, primer_len,
                                 n_free, discard, seed):
    """predicted_bifurcation (training.py) with a selectable primer source."""
    train_rhos = np.array([rho for _, rho in segments])
    j_central = int(np.argmin(np.abs(train_rhos - np.median(train_rhos))))
    j_lo = int(np.argmin(train_rhos))
    j_hi = int(np.argmax(train_rhos))
    lo_train, hi_train = train_rhos.min(), train_rhos.max()

    def primer_for(rho):
        if primer_mode == "none":
            return None
        if primer_mode == "central":
            j = j_central
        elif primer_mode == "edge":
            if rho < lo_train:
                j = j_lo
            elif rho > hi_train:
                j = j_hi
            else:
                j = int(np.argmin(np.abs(train_rhos - rho)))
        else:  # nearest (baseline)
            j = int(np.argmin(np.abs(train_rhos - rho)))
        return esn.standardize(segments[j][0][-primer_len:])

    zmax_by_rho, classes = [], []
    for rho in rho_grid:
        tr = esn.cold_extrapolate(rho, n_free=n_free, discard=discard, seed=seed,
                                  primer_hat=primer_for(rho))
        if tr is None or not np.all(np.isfinite(tr)) or np.max(np.abs(tr)) > 1e4:
            zmax_by_rho.append(np.empty(0)); classes.append("fixed_point"); continue
        zmax_by_rho.append(lorenz.z_maxima(tr[:, 2]))
        classes.append(lorenz.classify_trajectory(tr, lyap=None))
    return {"rho": np.asarray(rho_grid), "zmax": zmax_by_rho,
            "class": np.array(classes, dtype=object)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="row label for the log")
    ap.add_argument("--ref-lo", type=float, default=20.0)
    ap.add_argument("--ref-hi", type=float, default=36.0)
    ap.add_argument("--primer", default="nearest",
                    choices=["nearest", "central", "edge", "none"])
    ap.add_argument("--primer-len", type=int, default=1500)
    ap.add_argument("--gamma-p", type=float, default=0.1)
    ap.add_argument("--spectral-radius", type=float, default=0.6)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--n-free", type=int, default=4000)
    args = ap.parse_args()

    # knob (a): set the reference interval used by reservoir._normalize_param
    reservoir.RHO_REF = (args.ref_lo, args.ref_hi)

    cfg_over = dict(gamma_p=args.gamma_p, spectral_radius=args.spectral_radius)

    t0 = time.time()
    truth = load_truth()
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)
    preds = []
    for k in range(args.R):
        cfg = ESNConfig(seed=MASTER + k, **cfg_over)
        esn = training.train_realization(cfg, segments)
        preds.append(predicted_bifurcation_primed(
            esn, RHO_GRID, segments, args.primer, args.primer_len,
            args.n_free, 1200, MASTER + k))
    agg = training.aggregate_realizations(preds)
    m = training.c1_metrics(agg, truth)
    passed, parts = training.c1_pass(m)

    # normalized training span: how much of [-1,1] the training rho values fill
    pn = [reservoir._normalize_param(r) for r in RHO_TRAIN]
    train_fill = (min(pn), max(pn))

    row = {"label": args.label,
           "ref_interval": [args.ref_lo, args.ref_hi],
           "train_pnorm": [round(min(pn), 3), round(max(pn), 3)],
           "train_fill_frac": round((max(pn) - min(pn)) / 2.0, 3),
           "primer": args.primer, "primer_len": args.primer_len,
           "gamma_p": args.gamma_p, "spectral_radius": args.spectral_radius,
           "class_acc": float(m["class_acc"]),
           "rmse_frac": float(m["zmax_rmse_frac"]),
           "lyap_err": float(m["lyap_proxy_err"]),
           "n_wrong": int(m["n_class_wrong"]), "passed": bool(passed),
           "R": args.R, "n_free": args.n_free,
           "elapsed_s": round(time.time() - t0, 1)}

    log = {"log": []}
    if os.path.exists(LOG):
        with open(LOG) as f:
            log = json.load(f)
    log["log"].append(row)
    with open(LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[diag {args.label}] ref={args.ref_lo,args.ref_hi} primer={args.primer} "
          f"len={args.primer_len} | class={m['class_acc']*100:.1f}% "
          f"rmse={m['zmax_rmse_frac']*100:.2f}% lyap={m['lyap_proxy_err']*100:.1f}% "
          f"| {'PASS' if passed else 'fail'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
