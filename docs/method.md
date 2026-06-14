# The TIMBERS method

TIMBERS (Time-Integrated Marine Bézier Evolve and Refine Strategy) extends the BERS
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
