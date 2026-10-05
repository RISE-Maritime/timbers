# TiMBERS

**Time-Modulated Bézier Evolve and Refine Strategy** — GPU weather routing
that co-optimizes route geometry and an explicit speed profile.

TiMBERS extends the BERS reference method (arXiv:2605.31533) for deterministic
ship weather routing on gridded weather (e.g. ERA5 reanalysis). A route is a
degree-(K−1) Bézier curve with endpoints fixed at the ports; where BERS searches
geometry only (speed implicit, uniform time per curve parameter), TiMBERS
searches geometry **jointly with a time-allocation profile** — `n_speed`
log-weights, interpolated to the track segments and exp-normalized into
per-segment durations that sum to the fixed passage time. Speed allocation turns
out to be the first-order energy lever; global geometry is not.

> **Bring your own power model and cases.** This is a *method* library: the
> optimizer, the GPU sep-CMA-ES, the differentiable cost, the gradient polish,
> the land mask, the ERA5 loader and uncertainty-aware objectives (forecast
> ensembles and a forecast-error surrogate). It does **not**
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
5. **Uncertainty-aware objectives** (`timbers.ensemble`). The route cost
   evaluated over an ensemble: a real forecast ensemble (fields with a member
   axis and non-uniform forecast steps, e.g. ECMWF ENS), a forecast-error
   surrogate built by perturbing one field (spatial/temporal shift + amplitude
   scale, `perturbation_grid`), or both. Four objectives differ only in how
   members are reduced: cost from the nominal member or the member mean, safety
   from the nominal member or an ensemble constraint (`deterministic`,
   `expected_value`, `chance_constrained`, `joint`; the ensemble constraint is a
   breach probability, a CVaR, or the mean exceedance penalty).
   `score_members` gives per-member outcomes of a fixed route (its fragility,
   under the surrogate) and `member_series` per-segment values along a timed
   track.
6. **Re-planning** (`timbers.replan`). `sail_with_replanning` plans, sails one
   forecast cycle on the verifying weather under the power ceiling, and plans
   again from the realised position with the next forecast, keeping the
   original arrival time. The forecast is any callable returning a
   `model.Grids`, so archived ensembles, a single forecast and the verifying
   field itself (hindsight) all fit. `optimizer.fit_theta_to_track` projects a
   timed track onto the Bezier family, to seed the optimizer or to measure
   whether one curve can represent it.

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

`examples/compare_bers.py` isolates the explicit-speed lever — TiMBERS contains
BERS as the `n_speed = 0` (uniform-speed, geometry-only) special case, so the
same code path gives both. It prints a 2×2 ablation (uniform vs explicit speed ×
Stage 1 only vs + polish) on a storm scenario; see
[docs/method.md](docs/method.md) § *TiMBERS vs BERS*.

`examples/run_risk.py` demonstrates the forecast-error surrogate
(`timbers.ensemble`): it optimizes a deterministic and a robust route for the
same storm departure, then scores both across a forecast-error surrogate
ensemble — showing the robust route trade a little nominal energy for a much
lower chance of exceeding the wave limit.

Tests: `pytest`. The suite is data-free — unit invariants plus an end-to-end run
of the optimizer, the JAX evaluator, the host scorer, and the `solve_corridor`
backend, all on synthetic grids with the toy power model.

## Using your own problem

- **Weather**: load gridded NetCDF with `timbers.weather.load_era5`, or build the
  grid dicts directly (see `examples/run_toy.py`). The device path takes them as
  `timbers.model.Grids`: `Grids.from_era5(wind, wave)` for a single field, or
  `Grids(wind, wave, steps)` for fields with a member axis.
- **Power model**: implement `power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`.
  The device path (`timbers.model`/`timbers.optimizer`) calls it on JAX arrays;
  the host scorer (`timbers.scoring`) on NumPy arrays — `solve_corridor` and
  `make_polisher` take both (`power_fn`, `power_fn_host`).
- **Scoring**: `timbers.scoring.evaluate_route` and `evaluate_route_full`
  score a route on its planned schedule. `evaluate_route_saturated` sails it
  forward under a shaft-power ceiling: where the weather demands more power
  than the ceiling, the ship slows down and arrives late, so arrival time is an
  outcome rather than an input. `max_hours` and `t_offset_h` let a voyage be
  sailed in pieces, for example between re-plans. `v_max_for_power` gives the
  largest speed within the ceiling for given weather.
- **Corridor**: an `optimizer.Corridor` (port endpoints in a continuous
  working-longitude frame + passage time) and an optional land mask from
  `timbers.land.build_mask`. Pass the mask through
  `timbers.land.exclusion_raster` to add further exclusions (shallow water, a
  `domain` box kept inside the weather grid) and a ramp that grows with
  distance from open water, so a route that strays inland is pushed back out.
  Land is a hard constraint by default (`Penalty.lambda_land = 1e6`), and
  `load_era5` fills land-masked wave cells from the nearest sea cell
  (`land_fill="nearest"`) rather than with zero, which would make land read as
  calm water.

## Attribution

- **BERS** — the reference method TiMBERS extends. Daniel Precioso, Francisco
  Suárez, Javier Jiménez de la Jara, Rafael Ballester-Ripoll, David Gómez-Ullate,
  *BERS: Locally Optimal Continuous Algorithm for Maritime Weather Routing with
  Just-in-Time Arrival* (Bézier Evolve and Refine Strategy), arXiv:2605.31533
  (IE University; Universidad de Cádiz). TiMBERS reproduces the BERS baseline
  before extending it.
- **sep-CMA-ES** — Ros & Hansen (2008), *A Simple Modification in CMA-ES
  Achieving Linear Time and Space Complexity*.

## Data sources

TiMBERS bundles no data. If you use the loaders/scripts:

- **ERA5** reanalysis — Copernicus Climate Change Service (C3S) / ECMWF;
  downloaded by the user under the C3S licence (used by `timbers.weather`).
- **Natural Earth** land polygons — public domain (fetched by
  `scripts/download_natural_earth.py`, used by `timbers.land`).

## License

Apache-2.0 (see [LICENSE](LICENSE)).
