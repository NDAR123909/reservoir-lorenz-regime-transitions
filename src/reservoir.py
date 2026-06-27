"""
reservoir.py
============
The parameter-aware echo state network of Kong, Fan, Grebogi & Lai (2021), with
the locked architecture of methodology section 1.

State update (1.1):
    r(t+dt) = (1 - alpha) r(t) + alpha tanh( Wr r(t) + Win u(t) + b )

Input (1.2):
    u(t) = [ x_hat, y_hat, z_hat, p_hat ]
The three Lorenz coordinates are standardized with statistics pooled across the
whole training set; the parameter channel p_hat is rho mapped linearly onto the
fixed reference interval [20, 36]. State columns of Win carry the scaling
gamma_in, the parameter column carries gamma_p, kept separate on purpose (1.2).

Readout (1.3): ridge regression, closed form,
    Wout = Y R^T ( R R^T + lambda I )^-1

Locked hyperparameters (v2, 1.5): N=500, degree 6, spectral radius 0.6, leak 1.0,
gamma_in 0.10, gamma_p 0.1, bias 0.10, ridge 1e-6, washout 1000. (Spectral radius
and gamma_p were set in the C1 gate; their section-1.5 priors were 0.9 and 0.5.)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

# locked reference interval for the parameter channel (methodology 2.4)
RHO_REF = (20.0, 36.0)


@dataclass
class ESNConfig:
    # Post-gate LOCKED values (methodology v2, section 1.5). The C1 gate closed
    # on gamma_p=0.1 and spectral_radius=0.6; everything else stayed at the
    # section-1.5 priors. These carry into C2-C4 unchanged -- do not reopen.
    N: int = 500
    degree: int = 6                 # average in-degree of Wr
    spectral_radius: float = 0.6    # LOCKED in the C1 gate (prior was 0.9)
    leak: float = 1.0               # alpha
    gamma_in: float = 0.10          # state input scaling
    gamma_p: float = 0.1            # LOCKED in the C1 gate (prior was 0.5)
    bias_scale: float = 0.10
    ridge: float = 1e-6             # lambda
    washout: int = 1000
    n_inputs: int = 3               # x, y, z
    n_outputs: int = 3
    seed: int = 0                   # per-realization reservoir seed


def _normalize_param(rho: float) -> float:
    """Map rho onto the reference interval [20,36] -> [-1, 1]."""
    lo, hi = RHO_REF
    return 2.0 * (rho - lo) / (hi - lo) - 1.0


class ParameterAwareESN:
    """
    A single reservoir realization. Reservoir matrices depend only on the seed
    (methodology 5: realizations differ only in the random draws of Wr, Win, b).
    Pooled standardization statistics are set at fit time.
    """

    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        self._build_reservoir()
        self.Wout = None
        self.mu = np.zeros(cfg.n_inputs)     # pooled mean of x,y,z
        self.sd = np.ones(cfg.n_inputs)      # pooled std of x,y,z

    # ---- construction ---------------------------------------------------- #
    def _build_reservoir(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        N = cfg.N

        # sparse reservoir matrix: average in-degree = degree, density = degree/N
        density = cfg.degree / N
        W = sparse.random(N, N, density=density, format="csr",
                          random_state=rng,
                          data_rvs=lambda n: rng.uniform(-1.0, 1.0, n))
        # rescale to the target spectral radius.
        # ARPACK (eigs) seeds its starting residual from an UNSEEDED RNG when v0
        # is left to default, so the converged |lambda_max| wobbles at the ~1e-4
        # level from process to process. That wobble feeds straight into the
        # rescale factor and is enough to flip an occasional VPT threshold step,
        # which is the difference between a study that re-runs bit-for-bit and one
        # that only re-runs statistically. Handing eigs a deterministic v0 (drawn
        # from a generator keyed to cfg.seed, separate from rng so the W/Win/b
        # draw order is untouched) pins |lambda_max| to a fixed value for a given
        # seed. This changes no hyperparameter and no architecture -- the target
        # spectral_radius is unchanged; only the numerical path to it is made
        # reproducible (C5 determinism, methodology v2 section 1.5 untouched).
        v0 = np.random.default_rng(cfg.seed + 777).standard_normal(N)
        try:
            vals = eigs(W, k=1, which="LM", return_eigenvectors=False,
                        maxiter=1000, v0=v0)
            sr = abs(vals[0])
        except Exception:
            sr = max(abs(np.linalg.eigvals(W.toarray())))
        if sr > 0:
            W = W * (cfg.spectral_radius / sr)
        self.Wr = W.tocsr()

        # input matrix: 4 columns [x, y, z, p]; state columns scaled by gamma_in,
        # parameter column scaled by gamma_p (methodology 1.2)
        Win = rng.uniform(-1.0, 1.0, size=(N, cfg.n_inputs + 1))
        Win[:, :cfg.n_inputs] *= cfg.gamma_in
        Win[:, cfg.n_inputs] *= cfg.gamma_p
        self.Win = Win

        self.b = rng.uniform(-1.0, 1.0, size=N) * cfg.bias_scale

    # ---- standardization -------------------------------------------------- #
    def set_pooled_stats(self, mu, sd):
        self.mu = np.asarray(mu, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.sd[self.sd == 0] = 1.0

    def standardize(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.mu) / self.sd

    def destandardize(self, xyz_hat: np.ndarray) -> np.ndarray:
        return xyz_hat * self.sd + self.mu

    # ---- reservoir drive -------------------------------------------------- #
    def _update(self, r, u):
        cfg = self.cfg
        pre = self.Wr.dot(r) + self.Win.dot(u) + self.b
        return (1.0 - cfg.leak) * r + cfg.leak * np.tanh(pre)

    def drive(self, xyz_hat: np.ndarray, p_hat: float,
              r0: np.ndarray | None = None) -> np.ndarray:
        """
        Drive the reservoir teacher-forced through a standardized trajectory at
        fixed p_hat. Returns the full reservoir-state matrix (T, N), including
        washout (the caller slices it off).
        """
        T = xyz_hat.shape[0]
        r = np.zeros(self.cfg.N) if r0 is None else r0.copy()
        R = np.empty((T, self.cfg.N))
        for t in range(T):
            u = np.empty(self.cfg.n_inputs + 1)
            u[:self.cfg.n_inputs] = xyz_hat[t]
            u[self.cfg.n_inputs] = p_hat
            r = self._update(r, u)
            R[t] = r
        return R

    # ---- training (methodology 1.3) -------------------------------------- #
    def fit(self, segments):
        """
        Fit the linear readout by ridge regression over several training
        segments. Each segment is (xyz, rho): a raw Lorenz trajectory on the ESN
        grid and its constant rho. The reservoir is reset and a washout transient
        discarded between segments (methodology 1.2).

        Standardization statistics are pooled across all segments first.
        """
        cfg = self.cfg
        # pooled standardization (methodology 1.2 / 2): one mu, sd for all rho
        all_xyz = np.concatenate([seg[0] for seg in segments], axis=0)
        self.set_pooled_stats(all_xyz.mean(axis=0), all_xyz.std(axis=0))

        R_blocks, Y_blocks = [], []
        for xyz, rho in segments:
            xyz_hat = self.standardize(xyz)
            p_hat = _normalize_param(rho)
            R = self.drive(xyz_hat, p_hat)                 # (T, N)
            # one-step-ahead targets: predict next standardized state
            R_use = R[cfg.washout:-1]                       # states r(t)
            Y_use = xyz_hat[cfg.washout + 1:]               # targets u_state(t+1)
            R_blocks.append(R_use)
            Y_blocks.append(Y_use)

        R = np.concatenate(R_blocks, axis=0)                # (sum T, N)
        Y = np.concatenate(Y_blocks, axis=0)                # (sum T, 3)

        # augment with a constant term (linear readout with bias)
        Raug = np.hstack([R, np.ones((R.shape[0], 1))])     # (M, N+1)
        # Wout = Y^T Raug (Raug^T Raug + lambda I)^-1   ->  solve normal equations
        A = Raug.T @ Raug
        A[np.diag_indices_from(A)] += cfg.ridge
        B = Raug.T @ Y
        self.Wout = np.linalg.solve(A, B).T                 # (3, N+1)
        return self

    # ---- readout ---------------------------------------------------------- #
    def _readout(self, r):
        raug = np.concatenate([r, [1.0]])
        return self.Wout @ raug

    # ---- prediction protocols (methodology 1.4) -------------------------- #
    def warmup_then_freerun(self, warmup_xyz: np.ndarray, rho: float,
                            n_free: int) -> np.ndarray:
        """
        Teacher-force a short ground-truth segment at rho to set the reservoir
        state, then free-run closed-loop for n_free steps. Returns the predicted
        trajectory (n_free, 3) in *raw* (de-standardized) coordinates.
        """
        p_hat = _normalize_param(rho)
        warm_hat = self.standardize(warmup_xyz)
        r = np.zeros(self.cfg.N)
        # teacher-forced warmup
        for t in range(warm_hat.shape[0]):
            u = np.append(warm_hat[t], p_hat)
            r = self._update(r, u)
        # closed-loop free run
        pred = np.empty((n_free, self.cfg.n_outputs))
        y_hat = self._readout(r)
        for t in range(n_free):
            pred[t] = y_hat
            u = np.append(y_hat, p_hat)
            r = self._update(r, u)
            y_hat = self._readout(r)
        return self.destandardize(pred)

    def cold_extrapolate(self, rho: float, n_free: int,
                         n_warm: int = 200, discard: int = 1000,
                         seed: int | None = None,
                         primer_hat: np.ndarray | None = None) -> np.ndarray:
        """
        Cold parameter extrapolation (methodology 1.4): no ground-truth segment
        at rho. The reservoir is warmed from a generic initial state and then
        free-runs on p_hat(rho) alone. Returns the predicted trajectory in raw
        coordinates with the opening `discard` transient removed.

        The warmup is *generic* with respect to the target rho: it never uses a
        ground-truth trajectory at rho*. If `primer_hat` is given (a standardized
        trajectory taken from an available training rho), it is teacher-forced
        while the parameter channel is already held at p_hat(rho*), which primes
        the reservoir onto the Lorenz manifold and away from spurious basins
        before the free run. This is the multistability guard of methodology 9;
        residual single-reservoir basin artifacts are removed by the median over
        realizations (methodology 5, 3.5).
        """
        p_hat = _normalize_param(rho)
        r = np.zeros(self.cfg.N)

        if primer_hat is not None:
            for t in range(primer_hat.shape[0]):
                r = self._update(r, np.append(primer_hat[t], p_hat))
            y_hat = self._readout(r)
        else:
            rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
            y_hat = rng.normal(0.0, 1.0, size=self.cfg.n_outputs)
            for _ in range(n_warm):
                r = self._update(r, np.append(y_hat, p_hat))
                y_hat = self._readout(r)

        total = n_free + discard
        pred = np.empty((total, self.cfg.n_outputs))
        for t in range(total):
            pred[t] = y_hat
            r = self._update(r, np.append(y_hat, p_hat))
            y_hat = self._readout(r)
            if not np.all(np.isfinite(y_hat)) or np.max(np.abs(y_hat)) > 1e3:
                pred[t + 1:] = pred[t]
                break
        return self.destandardize(pred[discard:])
