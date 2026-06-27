"""
metrics.py
==========
Evaluation kernels of methodology section 3.

  - VGTT  (3.1): valid ground-truth time. Integrate the true trajectory twice
                 (RK4 at h and at h/2); the time at which they diverge by the
                 prediction threshold eps caps every reported prediction time.
  - VPT   (3.2): valid prediction time. First time the normalized error exceeds
                 eps = 0.4, reported in Lyapunov times, capped at the VGTT.
  - class (3.3): qualitative attractor class from a cold-extrapolation run.
  - D2    (3.4): Grassberger-Procaccia correlation dimension.
  - z-max (3.4): successive z-maxima distribution + Wasserstein distance.

Lyapunov time at rho=28: tau = 1/0.906 ~ 1.10 time units ~ 55 ESN steps.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance

import lorenz

EPS = 0.4                 # prediction error threshold (methodology 3.2)
DT_ESN = lorenz.H * lorenz.ESN_SUBSAMPLE   # 0.02
LAMBDA_MAX_28 = 0.906     # largest Lyapunov exponent at rho=28
TAU_LYAP_28 = 1.0 / LAMBDA_MAX_28          # ~1.10 time units


def lyap_time(rho: float = 28.0) -> float:
    """Lyapunov time in Lorenz time units; defaults to the rho=28 value."""
    if abs(rho - 28.0) < 1e-9:
        return TAU_LYAP_28
    le = lorenz.largest_lyapunov(rho, t_total=300.0)
    return 1.0 / le if le > 1e-6 else np.inf


# --------------------------------------------------------------------------- #
# 3.1  Valid ground-truth time                                                #
# --------------------------------------------------------------------------- #
def valid_ground_truth_time(rho: float, x0: np.ndarray, n_esn_steps: int,
                            eps: float = EPS) -> float:
    """
    Integrate the same trajectory twice (RK4 at h and at h/2), compare on the
    ESN grid, and return the time (in Lorenz units) at which the normalized
    error first exceeds eps. Beyond this the trajectory is no longer trustworthy.
    """
    n_rk4 = n_esn_steps * lorenz.ESN_SUBSAMPLE
    coarse = lorenz.integrate(rho, n_rk4, h=lorenz.H, x0=x0,
                              transient=0, subsample=lorenz.ESN_SUBSAMPLE)
    fine = lorenz.integrate(rho, n_rk4 * 2, h=lorenz.H / 2, x0=x0,
                            transient=0, subsample=lorenz.ESN_SUBSAMPLE * 2)
    n = min(len(coarse), len(fine))
    coarse, fine = coarse[:n], fine[:n]
    scale = np.sqrt(np.mean(np.sum(fine**2, axis=1)))
    err = np.linalg.norm(coarse - fine, axis=1) / scale
    idx = np.argmax(err > eps)
    if err[idx] <= eps:        # never exceeded within the window
        return n * DT_ESN
    return idx * DT_ESN


# --------------------------------------------------------------------------- #
# 3.2  Valid prediction time                                                  #
# --------------------------------------------------------------------------- #
def valid_prediction_time(pred: np.ndarray, truth: np.ndarray,
                          rho: float = 28.0, eps: float = EPS,
                          vgtt: float | None = None,
                          in_lyap_times: bool = True) -> float:
    """
    First time E(t) = ||pred - truth|| / sqrt(<||truth||^2>) exceeds eps.
    Reported in Lyapunov times and capped at the VGTT (methodology 3.2).
    """
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    scale = np.sqrt(np.mean(np.sum(truth**2, axis=1)))
    err = np.linalg.norm(pred - truth, axis=1) / scale
    idx = np.argmax(err > eps)
    vpt_time = (n * DT_ESN) if err[idx] <= eps else idx * DT_ESN
    if vgtt is not None:
        vpt_time = min(vpt_time, vgtt)        # the cap
    if in_lyap_times:
        return vpt_time / lyap_time(rho)
    return vpt_time


# --------------------------------------------------------------------------- #
# 3.3  Qualitative class                                                      #
# --------------------------------------------------------------------------- #
def qualitative_class(traj: np.ndarray, lyap: float | None = None) -> str:
    """
    Class label ('fixed_point' / 'periodic' / 'chaotic') for a predicted
    attractor, reusing the lorenz classifier so ground truth and prediction are
    judged on the same rule.
    """
    return lorenz.classify_trajectory(traj, lyap=lyap)


def predicted_lyapunov_proxy(traj: np.ndarray) -> float:
    """
    Cheap proxy for the sign of the predicted largest Lyapunov exponent, used
    when only a predicted time series is available (no governing equations).
    Estimated from the divergence of nearby return-map points.
    Positive -> chaotic-like, near zero -> periodic, negative -> collapsing.
    """
    z = traj[:, 2]
    zmax = lorenz.z_maxima(z)
    if zmax.size < 10:
        return -1.0
    # spread of the return map is a stand-in: a tight cycle => ~0, broad => >0
    return float(np.std(np.diff(zmax)))


# --------------------------------------------------------------------------- #
# 3.4  Correlation dimension (Grassberger-Procaccia)                          #
# --------------------------------------------------------------------------- #
def correlation_dimension(traj: np.ndarray, n_points: int = 2000,
                          n_r: int = 20, seed: int = 0) -> float:
    """
    Grassberger-Procaccia D2 from the slope of log C(r) vs log r over the
    scaling region. Subsamples to n_points for tractability.
    """
    rng = np.random.default_rng(seed)
    X = traj
    if len(X) > n_points:
        idx = rng.choice(len(X), n_points, replace=False)
        X = X[idx]
    # pairwise distances (upper triangle)
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices_from(d, k=1)
    dist = d[iu]
    dist = dist[dist > 0]
    if dist.size == 0:
        return np.nan
    rmin, rmax = np.percentile(dist, 1), np.percentile(dist, 50)
    if rmin <= 0 or rmax <= rmin:
        return np.nan
    radii = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(dist < r) for r in radii])
    mask = C > 0
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(np.log(radii[mask]), np.log(C[mask]), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------- #
# 3.4  z-maxima distribution agreement                                        #
# --------------------------------------------------------------------------- #
def zmax_distribution(traj: np.ndarray) -> np.ndarray:
    return lorenz.z_maxima(traj[:, 2])


def zmax_wasserstein(pred_traj: np.ndarray, true_traj: np.ndarray) -> float:
    """Wasserstein distance between predicted and true z-maxima distributions."""
    p = zmax_distribution(pred_traj)
    t = zmax_distribution(true_traj)
    if p.size < 2 or t.size < 2:
        return np.inf
    return float(wasserstein_distance(p, t))


def climate_agreement(pred_traj: np.ndarray, true_traj: np.ndarray,
                      d2_tol: float = 0.15) -> dict:
    """
    Climate cross-check (methodology 3.4): correlation-dimension match within
    d2_tol and a small Wasserstein distance between z-maxima distributions.
    """
    d2_pred = correlation_dimension(pred_traj)
    d2_true = correlation_dimension(true_traj)
    wass = zmax_wasserstein(pred_traj, true_traj)
    d2_ok = np.isfinite(d2_pred) and np.isfinite(d2_true) and abs(d2_pred - d2_true) < d2_tol
    return {
        "d2_pred": d2_pred,
        "d2_true": d2_true,
        "d2_match": bool(d2_ok),
        "zmax_wasserstein": wass,
    }
