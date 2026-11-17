"""
Chained, resumable driver for the C2-C4 sweeps, built on
the same checkpoint-and-resume pattern the C1 gate and the C1 v2 re-validation
used: one (config, realization) cell per unit of work, each appended to disk the
moment it finishes, so a killed or timed-out process never loses more than the
cell in flight. Long single jobs do not survive this environment,
so the run is driven as many short bounded invocations.

    python run_sweep.py --mode run --sweep C2 --R 32 --max-cells 12
    ...                                      (repeat until all cells on disk)
    python run_sweep.py --mode finalize --sweep C2 --R 32

--workers N spreads a batch across N processes. Each cell depends only on its
spec and realization seed, so the cells come out the same either way.

Architecture is the post-gate ESNConfig. This driver
never sets a hyperparameter; it varies only the sampling strategy.
"""
import sys, os, json, time, pickle, argparse, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lorenz, training, sweep
from reservoir import ESNConfig

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
TRUTH_CACHE = os.path.join(DATA, "truth_cache.pkl")
SEG_DIR = os.path.join(DATA, "segcache")
L_TOTAL = 120_000
MASTER = 20260613                 # reservoir master seed (study-wide)
MASTER_IC = MASTER + 90000        # IC seed stream

_TRUTH = None
def truth():
    global _TRUTH
    if _TRUTH is None:
        with open(TRUTH_CACHE, "rb") as f:
            raw = pickle.load(f)
        # cast trajectories back to float64 for the metric math
        _TRUTH = {}
        for k, v in raw.items():
            _TRUTH[k] = dict(warm=np.asarray(v["warm"], float),
                             free=np.asarray(v["free"], float),
                             vgtt=v["vgtt"], lyap=v["lyap"], **{"class": v["class"]})
    return _TRUTH


def _seg_seed(spec):
    # deterministic per-spec IC seed (seeds logged, study reruns
    # bit-for-bit). hashlib rather than the salted built-in hash().
    import hashlib
    h = int(hashlib.md5(spec["id"].encode()).hexdigest(), 16) % 100000
    return MASTER_IC + h


def get_segments(spec):
    os.makedirs(SEG_DIR, exist_ok=True)
    path = os.path.join(SEG_DIR, f"{spec['id']}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    segs = training.build_segments(spec["train_rhos"], L_TOTAL,
                                   transient_time=80.0, ic_seed=_seg_seed(spec))
    with open(path, "wb") as f:
        pickle.dump(segs, f)
    return segs


def cell_path(sweep_name, spec_id, k):
    d = os.path.join(DATA, f"{sweep_name}_cells")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{spec_id}_r{k:02d}.pkl")


def _run_cell(job):
    sweep_name, spec, k = job
    segs = get_segments(spec)
    cfg = ESNConfig(seed=MASTER + k)         # locked defaults; only seed varies
    esn = training.train_realization(cfg, segs)
    if spec["direction"] == "up":
        cell = sweep.measure_up(esn, spec, truth())
    else:
        # prime cold runs from the nearest-rho training segment
        train_rhos = np.array([r for _, r in segs])
        j = int(np.argmin(np.abs(train_rhos - spec["lower_edge"])))
        primer = esn.standardize(segs[j][0][-1500:])
        cell = sweep.measure_down(esn, spec, None, primer)
    cell["spec_id"] = spec["id"]; cell["x"] = spec["x"]; cell["k"] = k
    path = cell_path(sweep_name, spec["id"], k)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cell, f)
    os.replace(tmp, path)
    return f"{spec['id']} r{k:02d}"


def run(sweep_name, R, max_cells, only, workers=1):
    specs = sweep.sweep_points(sweep_name)
    if only:
        specs = [s for s in specs if s["id"] == only]
    todo = [(sweep_name, s, k) for s in specs for k in range(R)
            if not os.path.exists(cell_path(sweep_name, s["id"], k))]
    batch = todo[:max_cells]
    # build segment caches up front so parallel workers never race to write one
    for spec in {job[1]["id"]: job[1] for job in batch}.values():
        get_segments(spec)
    t0 = time.time()
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            done = pool.imap_unordered(_run_cell, batch)
            for n, name in enumerate(done, 1):
                print(f"[{sweep_name}] {name} done ({time.time()-t0:.0f}s, "
                      f"{n}/{len(batch)})", flush=True)
    else:
        for n, job in enumerate(batch, 1):
            name = _run_cell(job)
            print(f"[{sweep_name}] {name} done ({time.time()-t0:.0f}s, "
                  f"{n}/{len(batch)})", flush=True)
    if len(todo) > len(batch):
        print(f"[{sweep_name}] batch cap reached; resume to continue", flush=True)
    else:
        print(f"[{sweep_name}] no remaining cells in this call (processed {len(batch)})",
              flush=True)


# finalize: aggregate cells to median + IQR Delta-rho, acceptance, figure       
def _load_cells(sweep_name, spec, R):
    cells = []
    for k in range(R):
        p = cell_path(sweep_name, spec["id"], k)
        if os.path.exists(p):
            with open(p, "rb") as f:
                cells.append(pickle.load(f))
    return cells


def _median_curve_drho_up(cells):
    """Delta-rho from the median-over-realizations curve."""
    grid = cells[0]["grid"]
    vpt_in_med = np.median([c["vpt_in"] for c in cells])
    n = 0
    for i in range(len(grid)):
        med_vpt = np.median([c["vpt"][i] for c in cells])
        votes = [c["pred_class"][i] for c in cells]
        maj_chaotic = sum(v == "chaotic" for v in votes) > len(votes) / 2
        truth_chaotic = cells[0]["true_class"][i] == "chaotic"
        if (med_vpt >= sweep.FRAC * vpt_in_med) and maj_chaotic and truth_chaotic:
            n += 1
        else:
            break
    return round(n * sweep.DELTA_RHO, 4)


def _median_curve_depth_down(cells):
    grid = cells[0]["grid"]
    lowest = None
    for i in range(len(grid)):
        votes = [c["pred_class"][i] for c in cells]
        truth_cls = cells[0]["true_class"][i]
        maj = max(set(votes), key=votes.count)
        if maj == truth_cls:
            lowest = grid[i]
        else:
            break
    if lowest is None:
        return 0.0
    return round(max(0.0, sweep.HOPF_RHO - float(lowest)), 4)


def _majority_collapse_rho(cells):
    """Highest rho, walking down, where more than half the networks predict fixed_point."""
    grid = cells[0]["grid"]
    for i in range(len(grid)):
        n_fp = sum(c["pred_class"][i] == "fixed_point" for c in cells)
        if n_fp > len(cells) / 2:
            return float(grid[i])
    return None


def _bootstrap_band(cells, estimator, B=2000, seed=12345):
    """
    Realization-spread IQR on the median-curve estimator: resample
    realizations with replacement, recompute the median-curve Delta-rho each time,
    and return (q1, q3) of the bootstrap distribution. The band is centred on the
    full-sample point estimate by construction, so it represents how much the
    reported Delta-rho would wobble with the reservoir draw.
    """
    rng = np.random.default_rng(seed)
    n = len(cells)
    vals = np.empty(B)
    idx_all = np.arange(n)
    for b in range(B):
        sub = [cells[i] for i in rng.choice(idx_all, n, replace=True)]
        vals[b] = estimator(sub)
    q1, q3 = np.percentile(vals, [25, 75])
    return float(q1), float(q3)


def finalize(sweep_name, R):
    specs = sweep.sweep_points(sweep_name)
    rows = []
    for spec in specs:
        cells = _load_cells(sweep_name, spec, R)
        if not cells:
            print(f"[{sweep_name}] {spec['id']}: no cells yet"); continue
        if spec["direction"] == "up":
            estimator = _median_curve_drho_up
            ylabel = "delta_rho_up"
            med_vpt_in = float(np.median([c["vpt_in"] for c in cells]))
            per = np.array([sweep.delta_rho_up(c, vpt_in=med_vpt_in) for c in cells])
            edge = float(cells[0]["edge"])
            diag = dict(upper_edge=edge, median_vpt_in=med_vpt_in,
                        ceiling_rho=round(edge + estimator(cells), 4),
                        per_sample_len=L_TOTAL // spec["M"])
        else:
            estimator = _median_curve_depth_down
            ylabel = "across_hopf_depth"
            per = np.array([sweep.across_hopf_depth(c) for c in cells])
            edge = float(cells[0]["edge"])
            cross = float(np.mean([sweep.crossed_hopf(c) for c in cells]))
            # None means still chaotic at the floor; counted, not dropped
            trans = [sweep.predicted_transition_rho(c) for c in cells]
            settled = [t for t in trans if t is not None]
            if settled:
                t_med, t_q1, t_q3 = (float(v) for v in np.percentile(settled, [50, 25, 75]))
            else:
                t_med = t_q1 = t_q3 = float("nan")
            coex = lorenz.LANDMARKS["attractor_coexistence"]
            fp_rhos = [float(r) for c in cells
                       for r, p in zip(c["grid"], c["pred_class"]) if p == "fixed_point"]
            diag = dict(lower_edge=edge, d_above_hopf=spec["x"],
                        floor=float(cells[0]["floor"]),
                        crossed_hopf_frac=cross,
                        n_settled=len(settled),
                        n_never_settled=len(trans) - len(settled),
                        pred_transition_rho_median=t_med,
                        pred_transition_rho_q1=t_q1, pred_transition_rho_q3=t_q3,
                        majority_collapse_rho=_majority_collapse_rho(cells),
                        fixed_point_above_coexistence=any(r > coex for r in fp_rhos),
                        highest_fixed_point_rho=max(fp_rhos) if fp_rhos else None,
                        transition_rho_per_realization=trans,
                        per_sample_len=L_TOTAL // spec["M"])
        point = estimator(cells)                     # point estimate
        q1, q3 = _bootstrap_band(cells, estimator)   # realization band
        rows.append(dict(id=spec["id"], x=spec["x"], xlabel=spec["xlabel"],
                         train_rhos=spec["train_rhos"], M=spec["M"],
                         n=len(cells), median=float(point), q1=q1, q3=q3,
                         iqr=float(q3 - q1), ylabel=ylabel, diagnostics=diag,
                         per_realization=per.tolist(),
                         per_realization_median=float(np.median(per))))
        print(f"[{sweep_name}] {spec['id']:8s} x={spec['x']:5.2f}  "
              f"{ylabel}={point:.2f}  bootstrap IQR=[{q1:.2f},{q3:.2f}]  "
              f"(n={len(cells)})", flush=True)
        if spec["direction"] == "down":
            print(f"{'':12s}collapse median {diag['pred_transition_rho_median']:.2f} "
                  f"[{diag['pred_transition_rho_q1']:.2f}, {diag['pred_transition_rho_q3']:.2f}]  "
                  f"never settled {diag['n_never_settled']}/{len(cells)}  "
                  f"majority from {diag['majority_collapse_rho']}  "
                  f"FP above 24.06: {diag['fixed_point_above_coexistence']}", flush=True)
    out = dict(sweep=sweep_name, R=R, master_seed=MASTER,
               L_total=L_TOTAL, frac=sweep.FRAC, delta_rho=sweep.DELTA_RHO,
               locked_cfg=dict(gamma_p=ESNConfig().gamma_p,
                               spectral_radius=ESNConfig().spectral_radius,
                               N=ESNConfig().N),
               points=rows)
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, f"{sweep_name.lower()}_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{sweep_name}] result -> figures/{sweep_name.lower()}_result.json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "finalize"], default="run")
    ap.add_argument("--sweep", choices=["C2", "C3", "C4", "C2clamp"], required=True)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=12)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "run":
        run(args.sweep, args.R, args.max_cells, args.only, args.workers)
    else:
        finalize(args.sweep, args.R)
