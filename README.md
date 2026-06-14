# TIMBERS

**Time-Modulated Bézier Evolve and Refine Strategy** — GPU weather routing
that co-optimizes route geometry and an explicit speed profile.

TIMBERS extends the BERS reference method (arXiv 2605.31533) for deterministic
ship weather routing on gridded weather (e.g. ERA5 reanalysis). A route is a
degree-(K−1) Bézier curve with endpoints fixed at the ports; where BERS searches
geometry only (speed implicit, uniform time per curve parameter), TIMBERS
searches geometry **jointly with a time-allocation profile** — `n_speed`
log-weights, interpolated to the track segments and exp-normalized into
per-segment durations that sum to the fixed passage time. Speed allocation turns
out to be the first-order energy lever; global geometry is not.

> **Bring your own power model and cases.** This is a *method* library: the
> optimizer, the GPU sep-CMA-ES, the differentiable cost, the gradient polish,
> the land mask, the ERA5 loader, and the risk-aware extension. It does **not**
> include any vessel performance model or routing cases — you inject a
> `power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW` and supply your own
> corridors/weather. A trivial toy model and a runnable demo are in
> [`examples/`](examples/) so the pipeline runs out of the box.

## Method, in one pass

1. **Stage 1 — joint global search.** Separable CMA-ES (Ros & Hansen 2008),
   GPU-native in JAX: the entire generation loop runs inside one `lax.scan`
   (zero host round-trips) and the ES state is `vmap`ed over
   (seeds × departures), so one dispatch solves a whole corridor best-of-N.
2. **Scorer-aligned cost.** The candidate track is resampled to the reference
   scorer's uniform time grid inside the differentiable cost, so the optimizer
   minimizes the *scored* quantity and the wave-height penalty sees the
   sub-segment peaks the scorer sees (safe edge-riding up to the Hs limit).
3. **Feasibility-aware selection.** A small, steep soft penalty herds the
   population against the Hs/TWS boundary; best-of-N seed restarts supply legal
   edge-riders; hard limits are imposed at *selection* time.
4. **Stage 2 — gradient polish.** Per-waypoint lateral offsets along the route
   normal, co-refined with the speed profile, under gentle Adam + curvature
   regularization, keeping the best scored-feasible iterate.
5. **Risk-aware extension** (`timbers.risk`). Perturbation-fragility scoring
   and a chance-constrained objective over a forecast-error surrogate ensemble
   (spatial/temporal shift + amplitude scale of the weather sampling).

Details, design rationale, and negative results: [docs/method.md](docs/method.md).

## Install

```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # CUDA 12
```

## Run the demo

```bash
python scripts/download_natural_earth.py   # public-domain land polygons (optional)
PYTHONPATH=examples python examples/run_toy.py
```

`examples/run_toy.py` wires the toy power model in `examples/toy_power.py` to a
small synthetic corridor and weather grid and runs the full pipeline (device
sep-CMA cost → host scorer + polish) with no external data.

`examples/compare_bers.py` isolates the explicit-speed lever — TIMBERS contains
BERS as the `n_speed = 0` (uniform-speed, geometry-only) special case, so the
same code path gives both. It prints a 2×2 ablation (uniform vs explicit speed ×
Stage 1 only vs + polish) on a storm scenario; see
[docs/method.md](docs/method.md) § *TIMBERS vs BERS*.

Tests: `pytest`. The suite is data-free — unit invariants plus an end-to-end run
of the optimizer, the JAX evaluator, the host scorer, and the `solve_corridor`
backend, all on synthetic grids with the toy power model.

## Using your own problem

- **Weather**: load gridded NetCDF with `timbers.era5.load_era5`, or build the
  grid dicts directly (see `examples/run_toy.py`).
- **Power model**: implement `power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`.
  The device path (`timbers.model`/`timbers.optimizer`) calls it on JAX arrays;
  the host scorer (`timbers.scoring`) on NumPy arrays — `solve_corridor` and
  `make_polisher` take both (`power_fn`, `power_fn_host`).
- **Corridor**: an `optimizer.Corridor` (port endpoints in a continuous
  working-longitude frame + passage time) and an optional land mask from
  `timbers.land.build_mask`.

## Attribution

- **BERS** — the reference method TIMBERS extends. Daniel Precioso, Francisco
  Suárez, Javier Jiménez de la Jara, Rafael Ballester-Ripoll, David Gómez-Ullate,
  *BERS: Locally Optimal Continuous Algorithm for Maritime Weather Routing with
  Just-in-Time Arrival* (Bézier Evolve and Refine Strategy), arXiv:2605.31533
  (IE University; Universidad de Cádiz). TIMBERS reproduces the BERS baseline
  before extending it.
- **sep-CMA-ES** — Ros & Hansen (2008), *A Simple Modification in CMA-ES
  Achieving Linear Time and Space Complexity*.

## Data sources

TIMBERS bundles no data. If you use the loaders/scripts:

- **ERA5** reanalysis — Copernicus Climate Change Service (C3S) / ECMWF;
  downloaded by the user under the C3S licence (used by `timbers.era5`).
- **Natural Earth** land polygons — public domain (fetched by
  `scripts/download_natural_earth.py`, used by `timbers.land`).

## License

Apache-2.0 (see [LICENSE](LICENSE)).
