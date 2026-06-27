"""
build_truth_cache.py
====================
Precompute the ground-truth cache the C2/C3 same-class VPT measurement needs:
for each rho on a master grid, a seeded ground-truth trajectory (warmup + free
run), its valid ground-truth time (VGTT, the methodology-3.1 cap), and its
qualitative class. The cache is deterministic from the seed stream, so it is not
shipped in the repo tarball -- it is rebuilt here and reused by every sweep cell.

Built once; resumable (skips rho already on disk). Single output file
data/truth_cache.pkl.
"""
import sys, os, pickle, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lorenz, metrics

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "..", "data", "truth_cache.pkl")
MASTER_IC = 20260613 + 90000          # separate IC seed stream (methodology 5)
WARM = 200
FREE = 3000
RHO_LO, RHO_HI, RHO_STEP = 24.0, 40.0, 0.1


def _ic(rho):
    # deterministic per-rho initial condition, logged seed stream
    rng = np.random.default_rng(MASTER_IC + int(round(rho * 10)))
    return rng.uniform(-15, 15, size=3)


def build():
    grid = np.round(np.arange(RHO_LO, RHO_HI + 1e-9, RHO_STEP), 2)
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    t0 = time.time()
    done = 0
    for rho in grid:
        key = round(float(rho), 2)
        if key in cache:
            continue
        x0 = _ic(key)
        traj = lorenz.integrate_esn_grid(key, WARM + FREE, transient_time=80.0, x0=x0)
        warm = traj[:WARM]
        free = traj[WARM:WARM + FREE]
        vgtt = metrics.valid_ground_truth_time(key, x0, FREE)
        le = lorenz.largest_lyapunov(key, t_total=250.0)
        cls = lorenz.classify_trajectory(free, lyap=le)
        cache[key] = dict(warm=warm.astype(np.float32),
                          free=free.astype(np.float32),
                          vgtt=float(vgtt), lyap=float(le), **{"class": cls})
        done += 1
        if done % 20 == 0:
            tmp = CACHE + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(cache, f)
            os.replace(tmp, CACHE)
            print(f"[truth] {len(cache)}/{len(grid)} rho cached "
                  f"({time.time()-t0:.0f}s)", flush=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, CACHE)
    print(f"[truth] complete: {len(cache)} rho on disk", flush=True)


if __name__ == "__main__":
    build()
