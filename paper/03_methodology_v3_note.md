# Methodology note (v2 → v3): the C2 width grid and the section-2.4 exclusion

| **Document** | 03_methodology_v3_note.pdf | **Status** | Addendum to 03_methodology_v2. C1 gate stays passed; hyperparameters stay locked; the C1 criterion is untouched. |

This is a logged amendment under the section-10 living-document clause. It changes one thing in the C2 sweep and nothing else. The gate and the architecture are not reopened.

## The issue

Section 4's C2 sweep varies training-range width at a fixed density of one sample per ρ-unit, centered on ρ = 29. At the widest width, W = 10, the nominal window is [24, 34], which places the lowest training sample at ρ = 24. That is below the subcritical Hopf at ρ = 24.74. Section 2.4 excludes training values below the Hopf on purpose, because the coexistence region there makes the asymptotic class depend on the initial condition. So the C2 grid as written in v2 contains one point, W = 10, that trains on a sample section 2.4 says should not be used. Every other C2 width already sits entirely above the Hopf. This is the tension flagged at the end of Session 7.

## The amendment

The C2 width grid is constrained so that no training window includes a sample below ρ = 24.74. For W = 10 this clamps the lowest sample to the Hopf: the training set becomes {25, 26, ..., 34}, ten samples rather than eleven, with the per-sample length adjusting so the total training length stays at the fixed 120,000 ESN steps. The window center, the upper edge, and the upward-walk geometry of the measurement are unchanged, so the clamped point is directly comparable to the original. The other four widths are already compliant and are not touched.

## What the amendment does to the result

Nothing, and that is the point. I ran the clamped W = 10 as a separate configuration (`C2_W10c`, sweep key `C2clamp`, result in `figures/c2clamp_result.json`) at the full R = 32 against the locked, deterministic pipeline. It returns Δρ = 1.50 with an absolute ceiling of ρ = 35.50, the same as the pre-amendment W = 10. The underlying cells are genuinely different runs, not a re-label: the median in-window VPT is 7.60 for the clamped window against 7.67 for the original, and the per-realization values differ cell by cell. The extrapolation distance lands in the same place regardless.

## Consequence for how C2 is reported

The earlier reading, that W = 10 was contaminated by its sub-Hopf sample and that capping the widths would recover a monotone trend, is withdrawn. The robustness check shows the opposite: remove the sub-Hopf sample and the W = 10 upturn is still there, unchanged. The non-monotonicity in C2 is a real feature of the sweep, not a training artifact. Extrapolation reach falls from W = 2 through a minimum at W = 8, then rises again at W = 10, while the absolute ceiling climbs monotonically across the whole grid.

So C2 is reported as non-monotone, with the clamped configuration standing as the evidence that the widest-window result survives the section-2.4 exclusion. The clamped point is the section-2.4-compliant W = 10 for any reader who wants the grid to obey the training rule exactly; the two give the same answer. The original W = 10 cells are kept in the repo as the pre-amendment record.
