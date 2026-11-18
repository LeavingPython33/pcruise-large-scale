# Large-Scale Car Cruise-Control Simulations

This repository contains a standalone Python simulation of 100 vehicles on a
single-lane ring road. It compares IDM human drivers with the specified
pcruise proportional-derivative controller and two alternative ACC settings.
The experiment script and its outputs are separate from the proposed modular
package structure below.

## Repository layout

```text
traffic_ring_simulation.py  Standalone simulation and experiment runner
traffic_straight_simulation.py  Finite open-road platoon experiment
configs/
  scenarios/       Scenario definitions and parameter sets
  vehicles/        Vehicle and actuator parameter sets
data/
  raw/              Immutable input data, if a scenario needs it
  processed/        Derived or normalized input data
docs/
  architecture/     Design notes and model assumptions
outputs/
  logs/             Per-run diagnostics and execution logs
  metrics/          Per-run and aggregated numeric results
  plots/            Charts generated from simulation results
scripts/            Future command-line entry points for experiment batches
src/
  pcruise/
    analysis/       Result aggregation and visualization
    controller/     Cruise-control logic and actuator constraints
    plant/          Longitudinal vehicle and actuator dynamics
    scenarios/      Scenario setup, disturbances, and initial conditions
    simulation/     Time stepping, single runs, and batch orchestration
tests/
  integration/      End-to-end checks across model components
  unit/             Component-level checks
```

The listed directories are intentionally empty. Their `.gitkeep` files are
directory markers only, not executable code.

## Simulation architecture

The current experiment is implemented in
[`traffic_ring_simulation.py`](./traffic_ring_simulation.py). It includes
vehicle assignment, IDM and ACC models, fuel-rate integration, CSV summaries,
and plots. The component folders below describe a possible future refactoring;
they are not currently implemented as a package.

## Setup and run (Windows PowerShell)

Install Python 3.10 or later, then run these commands from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\traffic_ring_simulation.py
```

The script writes four PNG plots and CSV summaries beneath
`outputs\traffic_simulation\`. To load calibrated VT-Micro matrices, pass
`--fuel-coefficients .\vt_micro_coefficients.json` if you have separate
positive- and negative-acceleration matrices. If PowerShell blocks activation, use
`.\.venv\Scripts\python.exe` in place of `python` in the final two commands.

Fuel and performance runs use a per-vehicle distance window rather than a
fixed-duration measurement: after a 0.5 km warm-up, each vehicle contributes
the next 3 km of travel. The run ends when every vehicle completes that window;
elapsed simulation time can therefore differ by scenario and is recorded in the
run CSVs. Override the defaults with `--distance-km` and
`--warmup-distance-km`. `--duration` is the maximum allowed simulated time, not
the measurement horizon.

## Matched-speed fuel experiment

To compare pcruise fuel use without giving other controllers the same-density
advantage, run the separate equilibrium-initialized experiment:

```powershell
.\.venv\Scripts\python.exe .\traffic_fuel_efficiency_experiment.py --replicates 20
```

For each target speed, each controller starts at its own steady-state ring
spacing, then receives the same seeded braking-pulse profile. Fuel is measured
over an equal distance and compared at overlapping achieved mean speeds. The
script reports paired differences and 95% confidence intervals; negative
pcruise-minus-comparator values favor pcruise. It also reports whether the
tested speed sweep has a common flow range rather than extrapolating if it does
not. An unperturbed equilibrium baseline checks that the fuel rates agree at
equal speed before the disturbance comparisons. Results are saved under
`outputs\fuel_efficiency_experiment\`.

This is a controlled simulation experiment, not proof of real-vehicle fuel
savings. The supplied VT-Micro matrix is used for both acceleration signs;
calibrated sign-specific coefficients and real-world validation are needed
before interpreting the absolute fuel values as vehicle predictions. The
controller parameters are not adjusted to force a favorable result.

## Straight-road simulation

To run the non-circular, finite-platoon version, use:

```powershell
.\.venv\Scripts\python.exe .\traffic_straight_simulation.py --replicates 5
```

It writes separate results to `outputs\straight_road_simulation\`. The 100
vehicles follow a virtual lead on a straight, non-wrapping road; there is no
upstream inflow or vehicle re-entry, so this models a finite platoon rather than
a continuous-demand highway. The fundamental diagram estimates flow as initial
density times mean speed; it is not based on fixed-detector counts. The
`ride_smoothness.png` compares RMS longitudinal jerk and acceleration for ACC
vehicles during the lead-car braking pulse and recovery. These are rider-comfort
proxies, not a full comfort model: the simulation applies commanded acceleration
directly and does not model actuator lag or vertical vibration. Results are not
configured to guarantee that pcruise is smoother than the alternatives.
Fuel, speed, gap, and comfort metrics are measured over the same 3 km per
vehicle after a 0.5 km warm-up. Elapsed time varies and is reported in the run
CSV.

The same run creates a self-contained browser animation of 100 cars on the
straight road, comparing pcruise, aggressive ACC, and conservative ACC at 50%
equal placement. Open `outputs\straight_road_simulation\platoon_animation.html`
in a browser, or ask the script to open it when the run finishes:

```powershell
.\.venv\Scripts\python.exe .\traffic_straight_simulation.py --replicates 5 --open-visualization
```

The animation is a recorded run with play, pause, speed, and time controls; it
does not require a web server or internet connection.

## Mixed-traffic time-space diagram

Generate a high-resolution (300 DPI) time-space diagram for a separate
30-vehicle straight-road example:

```powershell
.\.venv\Scripts\python.exe .\traffic_time_space_simulation.py
```

The script writes `outputs\time_space_diagram.png`. It uses delayed IDM with
small stochastic speed-preference variation for human drivers and the specified
pcruise PD controller for ACC vehicles: `alpha=0.8`, `tau=2.5 s`, `g_min=2 m`,
`lambda=0.05`, and acceleration limits of `[-5, 4] m/s^2`. Human vehicles 23
and 8 brake toward 20 mph at 15 and 45 seconds. The diagram marks each braking
time and position. Human delay, noise, and IDM settings are illustrative
assumptions, not calibrated driver behavior.

To display the plot interactively as well as save it, add `--show`. NumPy and
Matplotlib are the only required packages and are listed in `requirements.txt`.

1. **Configuration** — vehicle and scenario parameters live separately under
   `configs/`. Each run should retain the exact configuration used.
2. **Scenario** — define initial conditions and optional changes such as target
   speed, road grade, or disturbances. A lead vehicle can be added later if the
   project scope includes adaptive cruise control.
3. **Plant** — model longitudinal vehicle behavior and relevant actuator limits.
4. **Controller** — compute a throttle/brake request from the target speed and
   measured vehicle state.
5. **Simulation** — advance the plant at a configured time step, record state
   and control signals, and support repeatable batches of independent runs.
6. **Analysis** — calculate performance and constraint metrics, aggregate
   batches, and save plots and tables under `outputs/`.

## Parameters and experiment status

The script defaults to 100 vehicles, a 0.1-second time step, a 180-second
simulation, a 0.5 km distance warm-up, a 3 km per-vehicle measurement window,
and five seeded placement runs per scenario. The simulation continues until
every vehicle completes the measurement distance, subject to a 1800-second
maximum. It sweeps 0%, 25%, 50%, 75%, and 100% ACC penetration; equal
interleaving and clustered placement; and five traffic densities. Fuel plots
show per-vehicle-km fuel, total fuel over the same fleet distance, average
speed, and elapsed time with approximate 95% confidence intervals. A
matched-speed diagnostic interpolates across the density sweep; density
therefore differs between controller curves. No
generated simulation results are currently committed.

The script uses the coefficient matrix supplied for this project by default,
for both positive and negative acceleration. It treats the polynomial output
as ln(mL/s), with speed in km/h and signed acceleration in km/h/s. The supplied
matrix's calibration vehicle and provenance are not validated by the script.
If separate matrices are available, pass a JSON file with numeric 4-by-4 arrays
named `positive_acceleration` and `negative_acceleration` using
`--fuel-coefficients`.

The per-vehicle fuel CSVs contain fuel burned during each vehicle's measured
distance window; fleet-normalized metrics use the sum of those equal-distance
windows.

Record measured runtime, hardware, software environment, and any memory or
parallelism limits when the first representative batch is run. There are no
runtime measurements, plots, or result files in this repository yet.

## Reproducibility and large batches

- Keep inputs and configuration distinct from generated outputs.
- Assign each run a stable identifier and retain its configuration, seed,
  software revision, and completion status with its results.
- Make batch execution bounded and restartable; avoid keeping all run traces in
  memory when writing results incrementally is practical.
- Keep large generated datasets and outputs out of version control unless the
  project specifically requires them.
- Add unit and integration checks as model components are implemented.

## Submission details

This scaffold was created in a workspace with no checked-out project commit.
After committing it, record the full repository revision for any submission;
do not treat this README or its directory markers as evidence that experiments
were run. A team plan and demonstration-video link have not been provided.
