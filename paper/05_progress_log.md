# Progress log — Session 6: methodology revision and C1 resolution

**Project:** Parameter-aware echo state networks for Lorenz regime transitions
**Author:** Noah Riego, Applied Physics, University of Arizona
**Date:** 14 June 2026
**Scope of this session:** resolve the blocked C1 gate through a logged methodology revision, re-validate, and lock the hyperparameters if it passes. The C2–C4 sweeps stay out of scope, same as last session.

## Where Session 5 left it

C1 was a partial pass and the gate was exhausted. Two of the three acceptance numbers cleared their bars comfortably: regime class at 100%, and Lyapunov/spread agreement at 6.3% against a 10% bar. The third did not. The z-maxima RMSE floored at about 7.49% of the z-range against a 5% bar, and I had walked the whole section-8 path (γ_p, spectral radius, ridge, N) without getting it under roughly 7.4% anywhere inside the locked ranges. The error was amplitude-only and it lived entirely in extrapolation, off the ends of the training set. The interior, where the model interpolates between training points, was tight.

Section 10 is clear about what that situation calls for: a logged methodology revision, re-issued before any sweep runs, not a hand-tune around the architecture the charter has fenced off. So this session is the revision. Best-found config going in was γ_p = 0.1, spectral radius = 0.6, ridge and N at their defaults.

## First, the two knobs that were not gate levers

Before touching the acceptance criterion I wanted to know whether the floor was actually fixed, or whether I had just not pulled the right non-architecture lever. The methodology leaves two encoding knobs outside the gate, and both were worth a bounded measurement: the parameter-channel reference interval (§2.4) and the cold-extrapolation primer (§1.4). I ran each variant the same way the gate ran, one config per process at R = 6, and logged everything to `data/diag_log.json`.

The reference interval went nowhere.

| reference interval | what it does | class | z-max RMSE | lyap |
|---|---|---|---|---|
| [20, 36] (baseline) | as-is | 100% | 7.71% | 5.5% |
| [18.81, 34.81] | recentre on the training midpoint, same width | 100% | 7.55% | 8.4% |
| [22.31, 31.31] | recentre and narrow, training fills ±0.5 | 100% | 7.64% | 7.2% |
| [12, 44] | widen, effective drive below the gate floor | 100% | 7.40% | 6.5% |

There is a reason recentring is the only interesting column there. The parameter drive into each node is γ_p · p̂, and p̂ = (ρ − centre)/half-width, so the per-node drive is (γ_p/half-width)·(ρ − centre). Narrowing the interval about its own centre is just dividing by a smaller half-width, which is the same as scaling γ_p — and the gate already swept γ_p. Only moving the centre is a genuinely new lever, and moving the centre did nothing. Widening past the gate floor (the [12,44] row, which is effectively γ_p below 0.1) shaved it to 7.40% and no further.

The primer went nowhere either.

| primer | class | z-max RMSE | lyap |
|---|---|---|---|
| nearest training trajectory (baseline) | 100% | 7.71% | 5.5% |
| none (generic warmup) | 100% | 7.53% | 6.5% |
| edge (boundary training trajectory) | 100% | 7.51% | 5.8% |

Dropping the primer entirely lands in the same place as keeping it. So the primer seats the reservoir on the right manifold, which is what it is for, but it has nothing to do with the amplitude at the edges. Both knobs sit between 7.40% and 7.71% no matter what I do to them. The floor is real and it is not an encoding choice.

## What the floor actually is

Splitting the error by ρ region is what cracked it. At the best config I broke the reconstruction into the interpolation interior and the two extrapolation edges, and added the chaotic band on its own.

| region | points | RMSE (mean z-max) | Wasserstein |
|---|---|---|---|
| low edge [24.00, 24.56) | 6 | 22.8% | 16.0% |
| interior [24.56, 29.06] | 45 | 5.5% | 1.0% |
| high edge (29.06, 32.00] | 30 | 1.2% | 1.2% |
| chaotic band [24.74, 32.00] | 73 | 2.3% | 1.0% |
| whole grid [24.00, 32.00] | 81 | 7.5% | 1.1% |

The whole failure is six grid points below the Hopf. The upper edge, which I had flagged in Session 5 as under-predicting the amplitude, is actually fine at 1.2% — that was me over-reading the scatter, and the numbers correct it. Everything above the Hopf is good. The damage is all in that sub-Hopf sliver, and the per-ρ values show why:

```
 rho   true_zmax   pred_zmax
24.00     23.41      32.82
24.10     33.26      32.91
24.20     23.30      33.15
24.30     33.36      33.73
24.40     23.62      33.57
24.50     33.97      33.92
24.60     24.11      33.90
24.70     34.44      34.04
```

The ground truth flips between roughly 23 and roughly 34 at neighbouring ρ. That is not the model missing the amplitude. That is the coexistence region the methodology already names in §2.4: below the Hopf the chaotic attractor and the stable C± pair both exist, and which one a trajectory lands on depends on the initial condition. The ground-truth diagram uses one initial condition per ρ, so it samples the two branches at random down there. The network sits steadily on the chaotic branch near 34, which is a perfectly reasonable thing to do, and gets charged a 20%-plus error for not matching a coin flip.

So §2.4 already excludes this region from training for exactly this reason. The C1 test grid was reaching into it anyway, and the amplitude metric was being read there. That is the bug, and it is in the metric's domain, not in the network.

## The revision (Option A) and why it is charter-legal

I scoped the C1 amplitude RMSE to the chaotic band ρ ≥ 24.74, the Hopf landmark. Regime-class accuracy still runs across the full grid, unchanged. The downward-across-Hopf behaviour is not dropped; it is what C4 measures, by qualitative-class match, which is the correct tool once the attractor type itself changes.

Three things make this legal under the charter rather than a goalpost shuffle. It changes no hyperparameter and adds no architecture. It introduces no new threshold — the 5% bar stays exactly where Session 4 put it. And it reads the amplitude metric only where the quantity it measures is single-valued, which is a fix to the metric's domain, not a loosening of it. I restrict by the landmark ρ value rather than by the data-driven class label on purpose: the classifier itself is fooled in the coexistence sliver (it calls those points chaotic from a single-IC Lyapunov estimate), so filtering on the label would not clean it up. The landmark does.

One implementation note. `c1_metrics` now takes `rmse_rho_min`, defaulting to the Hopf at 24.74, and still reports the full-grid number alongside it so the v1 floor stays on the record. Passing `rmse_rho_min=None` recovers the old metric exactly. Nothing about the predicted diagrams changed; only where I read the error did.

The methodology is re-issued as `03_methodology_v2.pdf` with the revision recorded at the top and §7, §1.5, §2.4, §3.4, §8, and §9 updated to match.

## Re-validation

Same model, revised criterion, R = 10 with the realizations checkpointed to disk one at a time (`run_c1_v2.py`, since long jobs still get killed here).

| criterion | result | threshold | verdict |
|---|---|---|---|
| regime-class accuracy (full grid) | 100.0% (0/81 wrong) | ≥ 95% | pass |
| z-maxima RMSE (ρ ≥ 24.74) | 2.31% of z-range | ≤ 5% | pass |
| Lyapunov / spread agreement | 5.6% | ≤ 10% | pass |

All three pass. For the record, the full-grid RMSE under the same run is 7.52%, which matches the Session-5 floor and confirms that the model did not change — only the evaluation domain did. The reconstruction is at `figures/fig1_c1_v2_bifurcation.png`, with the sub-Hopf coexistence band shaded and labelled as excluded from the amplitude metric. The result JSON is `figures/c1_v2_result.json`.

![C1 reproduction under methodology v2. Predicted z-maxima (red) against ground truth (grey) across rho in [24, 32]. The shaded band below the Hopf at rho approx 24.74 is the coexistence region excluded from the amplitude RMSE; dotted lines mark the four training rho. The fit is tight across the chaotic band.](../figures/fig1_c1_v2_bifurcation.png)

## Hyperparameters locked

The gate is closed, so these are fixed for C2–C4 and will not be reopened:

- reservoir size N = 500, average in-degree 6
- spectral radius = 0.6 (set in the gate)
- leak α = 1.0
- state input scaling γ_in = 0.10
- parameter scaling γ_p = 0.1 (set in the gate)
- bias scaling 0.10
- ridge λ = 1×10⁻⁶
- washout 1000, reference interval ρ ∈ [20, 36], L_total = 120,000, R = 32

## Status

- Methodology revised under §10 and re-issued as `03_methodology_v2.pdf`, change logged.
- Two non-architecture knobs measured and ruled out as the cause (`data/diag_log.json`); zone diagnostic in `data/diag_zones.json`.
- C1 re-validated at R = 10 and passing on all three criteria. Hyperparameters locked.
- C2–C4 are next session, against the locked architecture. The gate stays shut.
