"""
diag_zones.py
=============
Session 6 zone diagnostic. At the Session-5 best-found config, decompose the
C1 z-maxima error by region:
    low edge   : rho in [24.0, 24.56)   (downward extrapolation, toward/below Hopf)
    interior   : rho in [24.56, 29.06]  (interpolation between training points)
    high edge  : (29.06, 32.0]          (upward extrapolation)

For each zone it reports the RMSE on the per-rho mean z-maximum (the current C1
metric, as a fraction of the global z-range) and the median per-rho Wasserstein
distance between predicted and true z-maxima distributions (the methodology
section-3.4 climate metric), again as a fraction of the z-range.

This grounds the section-10 revision options in measured numbers rather than
assertion. R and n_free match the gate walk.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pickle
from scipy.stats import wasserstein_distance
import lorenz, training
from reservoir import ESNConfig

RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]
RHO_GRID = np.round(np.arange(24.0, 32.0 + 1e-9, 0.1), 2)
L_TOTAL = 120_000
MASTER = 20260613


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--n-free", type=int, default=4000)
    args = ap.parse_args()

    t0 = time.time()
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"ground_truth_{MASTER+777}.pkl")
    truth = pickle.load(open(cache, "rb"))
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)
    preds = []
    for k in range(args.R):
        cfg = ESNConfig(seed=MASTER + k, gamma_p=0.1, spectral_radius=0.6)
        esn = training.train_realization(cfg, segments)
        preds.append(training.predicted_bifurcation(
            esn, RHO_GRID, segments, n_free=args.n_free, discard=1200, seed=MASTER + k))
    agg = training.aggregate_realizations(preds)

    rho = truth["rho"]
    t_mean = np.array([np.mean(zm) if zm.size else np.nan for zm in truth["zmax"]])
    p_mean = agg["zmean"]
    all_true = np.concatenate([zm for zm in truth["zmax"] if zm.size])
    z_range = np.ptp(all_true)

    # per-rho Wasserstein between predicted-pool and true z-maxima distributions
    wass = np.full(len(rho), np.nan)
    for i in range(len(rho)):
        p, t = agg["zmax"][i], truth["zmax"][i]
        if p.size >= 2 and t.size >= 2:
            wass[i] = wasserstein_distance(p, t)

    # ground-truth regime class per rho, to locate the Hopf in the data
    tclass = truth["class"]
    zones = {
        "low_edge  [24.00,24.56)": (rho >= 24.0) & (rho < 24.56),
        "interior  [24.56,29.06]": (rho >= 24.56) & (rho <= 29.06),
        "high_edge (29.06,32.00]": (rho > 29.06) & (rho <= 32.0),
        "ALL       [24.00,32.00]": np.ones(len(rho), bool),
        "chaotic   [24.74,32.00]": (rho >= 24.74) & (rho <= 32.0),
        "chaotic-by-class (true) ": np.array([c == "chaotic" for c in tclass]),
    }
    print(f"\nz_range = {z_range:.2f}   (R={args.R}, n_free={args.n_free})")
    print(f"{'zone':<26}{'n':>4}{'RMSE-mean':>12}{'  Wass(med)':>13}")
    print("-" * 56)
    out = {}
    for name, mask in zones.items():
        v = mask & np.isfinite(p_mean) & np.isfinite(t_mean)
        rmse = np.sqrt(np.mean((p_mean[v] - t_mean[v])**2)) if v.sum() else np.nan
        wv = wass[mask & np.isfinite(wass)]
        wmed = np.median(wv) if wv.size else np.nan
        print(f"{name:<26}{int(mask.sum()):>4}{rmse/z_range*100:>10.2f}%{wmed/z_range*100:>11.2f}%")
        out[name] = {"n": int(mask.sum()),
                     "rmse_frac": float(rmse/z_range),
                     "wass_med_frac": float(wmed/z_range)}
    out["z_range"] = float(z_range)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
              "data", "diag_zones.json"), "w"), indent=2)

    print("\nper-rho, low end (rho <= 25.2):")
    print(f"{'rho':>6}{'true_zmax':>11}{'pred_zmax':>11}{'true_class':>13}{'pred_class':>13}")
    for i in range(len(rho)):
        if rho[i] <= 25.2:
            print(f"{rho[i]:>6.2f}{t_mean[i]:>11.2f}{p_mean[i]:>11.2f}"
                  f"{str(truth['class'][i]):>13}{str(agg['class'][i]):>13}")
    print(f"\n[zones] done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
