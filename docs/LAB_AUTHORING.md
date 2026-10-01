# Lab authoring guide (conventions every FL/EX lab follows)

This is the contract between the engine/app and each lab module. Reference implementations:
`src/convlab/labs/fl01_buck_boost.py` (switched converter, averaged model, independent RK45 path) and
`src/convlab/labs/ex02_coss_zvs.py` (nonlinear ODE with events, charge-coordinate RK4 path, energy ledger).

## Files

| What | Where |
|---|---|
| Lab module (physics + experiments + Korean content + `LAB`) | `src/convlab/labs/flNN_topic.py` or `exNN_topic.py` (auto-discovered; official IDs only) |
| Closed-form hand calculations used only for *comparison* | `src/convlab/reference/<topic>.py` (simulation code never imports these) |
| Tests | `tests/test_flNN.py` / `tests/test_exNN.py` |
| Shared helpers | `src/convlab/labs/_common.py` (ledger, decimation, bands, RK45 cross-check) |

Do not edit the engine, model, server or web files from a lab; if a lab needs a new generic feature, add it in the
lab module first and propose promotion.

## Physics rules (from the v4.0 instructions)

- SI internally. Show A/V/W/Hz/H/F/rad with scale; say phase-peak vs RMS vs DC, per-device vs per-leg vs total,
  per-module vs total, actual vs primary-referred. `n = Np/Ns`; referred to primary: `L' = n²L, C' = C/n², R' = n²R`.
- dq: amplitude-invariant, phase-peak; ω_e = p·ω_m. DAB: φ in rad, primary lead → positive HV→LV,
  `d = φ/π` is not `Δt/T = φ/2π`.
- Model level on every result: A analytic/FHA, B averaged dynamics, C ideal switching, D non-ideal commutation
  (synthetic). Never draw ripple on an averaged curve; never call an FHA-resistor sinusoidal circuit a switching
  rectifier verification; ideal-switch ZVS is `NOT_EVALUABLE`, a charge screen is `SCREEN_ONLY`.
- Losses: say what is included (channel, diode/dead time, E_sw, gate, Coss/recovery) and avoid double counting. A loss
  added to an ideal waveform afterwards is a *postprocessed loss estimate*; only losses inside the state equations make
  an energy balance "coupled".
- Inputs default to `ASSUMED`/`TEXTBOOK` synthetic values; `DATASHEET` needs exact part/revision/conditions/URL (none
  are used). Missing real data → `MISSING_INPUT` for the precise claim, but the synthetic learning model still runs.
- No coefficient tweaking to hit a golden number. If a textbook number is not reproduced, investigate, keep the
  physics, and write the finding for `docs/ERRATA.md` (id, where, original, corrected, evidence, impact).
- Keep seed failures (e.g. CLLC seed FAIL, CPL 100 µF unstable): they are results.
- Steady state: per-cycle state/energy criteria and/or shooting; record initial condition, DC offset, rectifier state,
  damping reasons. Averages/RMS exact over segments (no edge smoothing). Energy residual
  `E_in − E_out − E_loss − ΔW` in J, normalised by max(port energy, 1 % rated power × window).
- Different failures are different statuses (`model/status.py`): FAIL_CONSTRAINT, NO_SOLUTION, UNSTABLE,
  SOLVER_FAILED, OUT_OF_VALIDITY, MISSING_INPUT, NOT_EVALUABLE, NOT_RUN_ENVIRONMENT, UNRESOLVED_RANKING, …
- Inputs out of validity are rejected by `Param(vmin, vmax)` with a reason, never clamped.

## Verification rules

- Every key number is checked on at least one *independent* path: closed form vs piecewise/exact integration vs
  event/ODE solver vs a separately written formulation (different state variables, different integrator), energy
  identity, limiting case, tolerance/step convergence. A test that calls the same helper to produce both sides is a
  regression test and is labelled as such (`Check(independent=False)`).
- Textbook reference numbers are typed literally in tests (independent of `reference/`).
- Tolerances (textbook ch.19 starting targets): nominal ideal P ≤ 1 %, RMS ≤ 2 %, ripple ≤ 5 %; energy residual
  ≤ 0.5 % (learning gate; exact engines reach ~1e-12); step/tolerance halving: mean/RMS change ≤ 1 %, peak ≤ 2 %.

## Result content (what the UI shows)

- `Metric(key, label, value, unit, ref=, ref_label=, tol=, basis=, note=)` — basis says per-device/total/RMS/window.
- `Plot(..., proved=..., not_yet=...)` — every plot states in Korean what it proved and what it has not. Both are required: `convlab run-all` reports an empty one as a contract issue and exits 1 (CI fails).
  Time plots in the same `group` share the cursor; `bands` (state intervals) link to circuit `modes`.
- `Check(name, status, value, unit, threshold, path=, independent=, detail=)` for verification.
- `res.verdict(code, why)`; `res.assumptions`, `res.not_valid_for`, `res.interpretation` (Korean, "왜 그런가").
- Circuit: `model.circuit.Circuit` (elements with terminal a = reference-positive end, wires, probes, modes).

## Experiment content (the learning flow)

goal (이번에 배우는 것) → params + presets (nominal/reference/corner/failure) → `suggested` change (machine-readable) →
`Prediction(question, options, expected, why, metric_keys, handcalc)` → run → plots/circuit → student / expert text →
customer KO + EN → `Question`s (KO + EN, answers folded, must_include rubric points). Korean for user text; code and
comments in English. Never write that the learner is interview-ready; simulation PASS ≠ learning complete.

## Performance

Default preset of each experiment should run in ≲ 3 s (≲ 10 s for explicitly `runtime_hint="seconds"` sweeps).

## Running

```bash
python -m pytest -q tests/test_flNN.py
python -m convlab run FL08 sps_nominal --preset nominal
python -m convlab serve --port 8765 --no-browser
```
