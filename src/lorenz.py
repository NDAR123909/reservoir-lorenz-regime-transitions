"""
lorenz.py
=========
Lorenz 1963 system: fixed-step RK4 integrator, z-maxima return map, largest
Lyapunov exponent (Benettin), and the numerical bifurcation-map / regime
classifier specified in methodology section 2.3.

Conventions locked in 03_methodology.pdf:
    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z
with sigma = 10, beta = 8/3 held fixed; rho is the only axis the study moves.
Integration step h = 0.01. The ESN observes every second RK4 point, so its
native step is dt = 0.02.
"""

from __future__ import annotations
import numpy as np

SIGMA = 10.0
BETA = 8.0 / 3.0
H = 0.01          # RK4 step
ESN_SUBSAMPLE = 2  # ESN sees every 2nd point -> dt = 0.02


def _f(state: np.ndarray, rho: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (rho - z) - y,
        x * y - BETA * z,
    ])


def _jac(state: np.ndarray, rho: float) -> np.ndarray:
    """Jacobian of the Lorenz flow, used for the tangent-space LE."""
    x, y, z = state
    return np.array([
        [-SIGMA,   SIGMA,  0.0],
        [rho - z,  -1.0,   -x],
        [y,        x,      -BETA],
    ])


def rk4_step(state: np.ndarray, rho: float, h: float = H) -> np.ndarray:
    k1 = _f(state, rho)
    k2 = _f(state + 0.5 * h * k1, rho)
    k3 = _f(state + 0.5 * h * k2, rho)
    k4 = _f(state + h * k3, rho)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(rho: float,
              n_steps: int,
              h: float = H,
              x0: np.ndarray | None = None,
              transient: int = 0,
              subsample: int = 1,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Integrate the Lorenz system at a single rho.

    Returns an array of shape (n_kept, 3). `transient` RK4 steps are integrated
    and thrown away first; `subsample` keeps every k-th retained point (use
    ESN_SUBSAMPLE to get the dt = 0.02 ESN grid).
    """
    if x0 is None:
        if rng is None:
            x0 = np.array([1.0, 1.0, 1.0])
        else:
            x0 = rng.uniform(-15, 15, size=3)
    state = np.asarray(x0, dtype=float).copy()

    for _ in range(transient):
        state = rk4_step(state, rho, h)

    out = np.empty((n_steps, 3))
    for i in range(n_steps):
        state = rk4_step(state, rho, h)
        out[i] = state

    if subsample > 1:
        out = out[::subsample]
    return out


def integrate_esn_grid(rho: float, n_esn_steps: int, transient_time: float = 40.0,
                       x0=None, rng=None) -> np.ndarray:
    """
    Convenience wrapper that returns a trajectory on the ESN grid (dt = 0.02).
    `n_esn_steps` is the number of ESN-grid points returned. `transient_time`
    is in Lorenz time units.
    """
    transient_rk4 = int(round(transient_time / H))
    n_rk4 = n_esn_steps * ESN_SUBSAMPLE
    return integrate(rho, n_rk4, h=H, x0=x0,
                     transient=transient_rk4, subsample=ESN_SUBSAMPLE, rng=rng)


# --------------------------------------------------------------------------- #
# z-maxima return map                                                         #
# --------------------------------------------------------------------------- #
def z_maxima(z: np.ndarray) -> np.ndarray:
    """Successive local maxima of the z-coordinate (the Lorenz return map)."""
    z = np.asarray(z)
    if z.size < 3:
        return np.empty(0)
    interior = (z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])
    return z[1:-1][interior]


# --------------------------------------------------------------------------- #
# Largest Lyapunov exponent (Benettin, tangent space)                          #
# --------------------------------------------------------------------------- #
def largest_lyapunov(rho: float,
                     t_total: float = 1000.0,
                     t_transient: float = 100.0,
                     renorm_every: float = 1.0,
                     h: float = H,
                     x0: np.ndarray | None = None) -> float:
    """
    Largest Lyapunov exponent via the tangent-space method: co-evolve a tangent
    vector under the Jacobian, renormalize at fixed intervals, and average the
    log growth. Returned in units of 1/time.
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0])
    state = np.asarray(x0, dtype=float).copy()

    # discard transient on the trajectory itself
    for _ in range(int(round(t_transient / h))):
        state = rk4_step(state, rho, h)

    delta = np.array([1e-8, 0.0, 0.0])
    delta = delta / np.linalg.norm(delta) * 1e-8
    d0 = np.linalg.norm(delta)

    steps_per_renorm = max(1, int(round(renorm_every / h)))
    n_renorm = int(round(t_total / (steps_per_renorm * h)))
    log_sum = 0.0

    for _ in range(n_renorm):
        for _ in range(steps_per_renorm):
            # RK4 on the joint (state, tangent) system
            k1s = _f(state, rho);                 k1d = _jac(state, rho) @ delta
            s2 = state + 0.5 * h * k1s;           d2 = delta + 0.5 * h * k1d
            k2s = _f(s2, rho);                    k2d = _jac(s2, rho) @ d2
            s3 = state + 0.5 * h * k2s;           d3 = delta + 0.5 * h * k2d
            k3s = _f(s3, rho);                    k3d = _jac(s3, rho) @ d3
            s4 = state + h * k3s;                 d4 = delta + h * k3d
            k4s = _f(s4, rho);                    k4d = _jac(s4, rho) @ d4
            state = state + (h / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
            delta = delta + (h / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
        dnorm = np.linalg.norm(delta)
        log_sum += np.log(dnorm / d0)
        delta = delta * (d0 / dnorm)  # renormalize, keep direction

    return log_sum / (n_renorm * steps_per_renorm * h)


# --------------------------------------------------------------------------- #
# Regime classifier (methodology 2.3)                                          #
# --------------------------------------------------------------------------- #
# class thresholds (methodology 3.3): chaotic above +0.01, fixed point below
# -0.01, periodic in between; plus a variance-collapse test for fixed points.
LYAP_CHAOS = 0.01
LYAP_FP = -0.01


def classify_trajectory(traj: np.ndarray,
                        lyap: float | None = None,
                        collapse_std: float = 1e-2) -> str:
    """
    Classify an attractor as 'fixed_point', 'periodic', or 'chaotic' from a
    trajectory and (optionally) a precomputed largest Lyapunov exponent.

    The variance-collapse test catches fixed points whose LE estimate is noisy:
    if the spread of the z-coordinate over the (post-transient) window is tiny,
    the trajectory has settled onto an equilibrium.
    """
    z = traj[:, 2]
    zmax = z_maxima(z)

    # collapse test: trajectory has stopped moving
    if np.std(z) < collapse_std or zmax.size < 3:
        return "fixed_point"

    if lyap is not None:
        if lyap > LYAP_CHAOS:
            return "chaotic"
        if lyap < LYAP_FP:
            return "fixed_point"
        # in-between LE -> distinguish periodic from chaotic by the spread of
        # the return map: a periodic orbit visits only a few distinct maxima.
        if _few_distinct(zmax):
            return "periodic"
        return "chaotic"

    # no LE supplied: fall back on the return-map spread alone
    return "periodic" if _few_distinct(zmax) else "chaotic"


def _few_distinct(zmax: np.ndarray, tol_frac: float = 0.02, max_distinct: int = 8) -> bool:
    """True if the z-maxima collapse onto a small number of discrete values."""
    if zmax.size < 4:
        return True
    rng = np.ptp(zmax)
    if rng < 1e-6:
        return True
    tol = tol_frac * rng
    vals = np.sort(zmax)
    distinct = [vals[0]]
    for v in vals[1:]:
        if v - distinct[-1] > tol:
            distinct.append(v)
    return len(distinct) <= max_distinct


def bifurcation_map(rho_grid: np.ndarray,
                    n_esn_steps: int = 8000,
                    transient_time: float = 80.0,
                    lyap_t_total: float = 400.0,
                    with_lyap: bool = True,
                    rng: np.random.Generator | None = None) -> dict:
    """
    Numerical bifurcation map of methodology 2.3. For each rho on the grid:
    integrate past the transient, collect z-maxima, estimate the largest
    Lyapunov exponent, and classify the regime.

    Returns a dict with arrays keyed by 'rho', 'lyap', 'class', and a list
    'zmax' (the z-maxima at each rho, for the bifurcation diagram).
    """
    rho_grid = np.asarray(rho_grid, dtype=float)
    classes, lyaps, zmaxes = [], [], []
    for rho in rho_grid:
        traj = integrate_esn_grid(rho, n_esn_steps, transient_time=transient_time, rng=rng)
        zm = z_maxima(traj[:, 2])
        lyap = largest_lyapunov(rho, t_total=lyap_t_total) if with_lyap else None
        cls = classify_trajectory(traj, lyap=lyap)
        classes.append(cls)
        lyaps.append(lyap if lyap is not None else np.nan)
        zmaxes.append(zm)
    return {
        "rho": rho_grid,
        "lyap": np.array(lyaps),
        "class": np.array(classes, dtype=object),
        "zmax": zmaxes,
    }


# Known landmarks (methodology 2.3), used as a self-check on the classifier.
LANDMARKS = {
    "pitchfork": 1.0,
    "homoclinic_explosion": 13.93,
    "attractor_coexistence": 24.06,
    "subcritical_hopf": 24.74,
    "standard_chaos": 28.0,
}


if __name__ == "__main__":
    # quick smoke test: classify the canonical chaotic point and a fixed point
    for rho in (20.0, 28.0):
        tr = integrate_esn_grid(rho, 8000, transient_time=80.0)
        le = largest_lyapunov(rho, t_total=400.0)
        print(f"rho={rho:5.1f}  lyap={le:+.4f}  class={classify_trajectory(tr, le)}")
