# The TiMBERS method

TiMBERS (Time-Modulated Bézier Evolve and Refine Strategy) extends the BERS
reference method (arXiv 2605.31533) for deterministic ship weather routing on
gridded weather. BERS is reproduced as the baseline and then extended. This
document summarizes what changed, why, and — equally important — what was tried
and did *not* help.

## Baseline: what BERS does

Two-stage deterministic weather routing on reanalysis weather:

- **Stage 1** — CMA-ES over the interior control points of a degree-(K−1)
  Bézier curve (path geometry), endpoints fixed at the ports. Speed is
  **implicit**: uniform time per curve parameter (geometry only).
- **Stage 2 (FMS)** — local gradient/Newton refinement of the route vertices.

## 1. Core method changes

**A. Explicit speed optimization — the headline change.** A time-allocation
profile (`n_speed` log-weights → per-segment durations summing to the fixed
passage time, `timbers.optimizer.time_alloc`) is co-optimized *jointly with
geometry* in stage 1. The parameterization is a strict superset of BERS:
zero log-weights reproduce the implicit uniform-speed model exactly. This is
the **dominant energy lever**: `n_speed` is the dominant resolution axis in
sweep studies, and a dynamic-programming study independently shows speed
allocation is first-order while global geometry is not. It also makes BERS's
stage-2 FMS redundant.

**B. Scorer-aligned cost integration (`align_dt_h`).** The L-point track is
resampled to the reference scorer's uniform-time grid inside the cost, and
energy/Hs are integrated there — the optimizer minimizes the *actual scored
quantity*, and the Hs penalty sees the sub-segment peaks the scorer sees. This
enables safe edge-riding up to the wave-height limit.

**C. Feasibility-aware selection + boundary herding.** A small, steep soft
penalty (`hs_soft` margin, `aH`) herds the population against the Hs/TWS
boundary; best-of-N seed diversity supplies the legal edge-riders; selection
hard-gates against the application's Hs/TWS limits. This decouples optimizer
pressure from the hard constraint.

**D. Best-of-N seed restarts.** The energy landscape is multimodal
(north-of-storm vs south-of-storm vs GC-hugging basins); N restarts cut
per-departure variance.

**E. Gradient local-refinement polish (supersedes FMS).** Per-waypoint lateral
offset along the route normal + co-refined speed, gentle Adam, curvature
regularization, keeping the best *scored-feasible* iterate
(`timbers.polish`). This is the careful version of the CMA→gradient second
step that BERS's FMS represents (a naive version diverged); the original FMS
was redundant once speed is explicit. Additive, storm-concentrated, and
shown exhausted — a much heavier gradient stage adds a negligible increment.

**F. Per-case resolution + budget.** Bézier degree (`K`), speed resolution
(`n_speed`), population size and iteration budget are tuned per case; harder
corridors reward finer resolution and a larger budget that a fixed resolution
does not reach.

## 2. Infrastructure (enables the above at scale)

**G. Fully GPU/JAX differentiable cost** (`timbers.optimizer`,
`timbers.model`). The penalized cost for the whole CMA-ES population is
evaluated batched on the GPU; the cost is differentiable end-to-end.

**H. GPU-native separable CMA-ES** (`timbers.cmaes`). Replaces a CPU-bound
host loop (profiled: ask/tell dominated wall time, GPU idle) with a JAX
sep-CMA-ES (Ros & Hansen 2008): the **entire generation loop runs inside one
`lax.scan`** (zero host round-trips), and the ES state is `vmap`ed over
(seeds × departures) for one-dispatch best-of-N. Roughly an order of magnitude
faster per instance than the host full-CMA loop; diagonal CMA is
quality-neutral on this problem.

## 3. Negative results (what does *not* help)

These bound the search space:

- **Diverse topology / MAP-Elites seeding:** gross north/south topology is not
  the lever; the GC-seeded basin is already the good (storm-avoiding) route.
- **Local-control B-spline basis:** a richer basis needs a *matched* optimizer;
  CMA-ES from a straight init cannot coordinate it.
- **Time-dependent / global DP (isochrone-style):** a spatial-only DP *loses*
  to the speed-aware Bézier — global geometry is not the lever.
- **Heavier gradient polish:** multi-restart, much larger budget adds a
  negligible increment — the gradient stage is exhausted.

## 4. Risk-aware extension (beyond BERS's deterministic scope)

`timbers.risk` treats forecast error as a perturbation ensemble over the
weather sampling (spatial shift, temporal shift, amplitude scale) with the
route held fixed:

- **Perturbation-fragility** (`make_scorer`): re-score a fixed route under
  jittered weather, GPU-batched. Finding: routes optimized to ride just below
  the wave-height limit on a single reanalysis are fragile — they violate the
  limit with high probability under modest forecast error on storm departures.
- **Chance-constrained objective** (`make_robust_cost`): minimize *nominal*
  energy + ensemble-mean exceedance from the *hard* limit, so the safety
  buffer is set by the local forecast spread rather than a hand-tuned
  `hs_soft`. A real, tunable risk control where `hs_soft` is degenerate
  (trading a lower violation probability for some energy).

This reframes deterministic-routing spread as **method vs risk-appetite**.
`examples/run_risk.py` makes it concrete: it optimizes a deterministic and a
chance-constrained route for the same toy storm departure, then scores both
across an 18-member forecast-error surrogate ensemble (spatial ±0.3°, temporal
±3 h, Hs ×1.12) and reports the fraction whose max Hs crosses the 7 m limit:

```
route              nom MWh    nom maxHs   Hs>limit (ensemble)
deterministic       7869.7        4.29 m              17%
robust              9170.0        2.96 m               0%
```

The deterministic route is cheapest yet infeasible in ~1 in 6 perturbed
forecasts; the chance-constrained route eliminates that for a ~16% nominal-energy
premium — the safety buffer emerging from the local forecast spread rather than a
hand-tuned margin. Same caveat as §5: toy power model on a constructed scenario,
so the magnitudes are illustrative of the mechanism, not real-vessel numbers.

## 5. TiMBERS vs BERS, head to head

Because the time-allocation profile reduces to uniform speed at `n_speed = 0`,
TiMBERS *contains* BERS: the same code path, with speed switched off, is BERS's
implicit uniform-speed, geometry-only model (and with `n_speed = 0` the gradient
polish refines only the lateral geometry — a stand-in for BERS's FMS). So the
explicit-speed contribution can be isolated by toggling a single knob, with
geometry, optimizer, seeds and polish held identical.

`examples/compare_bers.py` runs this 2×2 ablation (uniform vs explicit speed ×
Stage 1 only vs + polish) on a synthetic corridor with a localized space-time
storm and the toy power model:

```
                        uniform speed (BERS)  explicit speed (TiMBERS)
Stage 1 only                        7977.772                  7867.575   (+1.4% from speed)
+ gradient polish                  ~7975.7                   ~7862      (+1.4% from speed)
```

(Representative run. The Stage-1 figures are deterministic; the + polish row
varies by ~0.02% run to run — the gradient stage is scored on the host and
inherits XLA nondeterminism — so treat its digits as approximate.)

The uniform→explicit-speed column delta is the TiMBERS lever. Two honest
caveats: (1) this is the **toy** power model on a constructed scenario, so the
*magnitude* is illustrative of the mechanism, not a real-vessel or benchmark
result; on benign weather the lever is near-zero — it only pays off when there
is weather structure to time the crossing around. (2) "BERS mode" is BERS's
*algorithm* reconstructed as the `n_speed = 0` special case, not a port of the
authors' code, and does not reproduce their paper's benchmark numbers.
