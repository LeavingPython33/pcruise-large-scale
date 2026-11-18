#!/usr/bin/env python3
"""Open straight-road platoon experiment; no circular wraparound or re-entry.

The 100 simulated vehicles form a finite platoon on an unbounded straight
coordinate axis. A non-counted virtual lead vehicle supplies the front
boundary condition and braking pulse. There is no upstream inflow, so this is
a finite-platoon experiment rather than a continuous-demand highway model.

Run from the repository root with:
    python traffic_straight_simulation.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional
import webbrowser

import matplotlib.pyplot as plt
import numpy as np

from traffic_ring_simulation import (
    ACC_VARIANTS,
    ACCParameters,
    DENSITIES_VEH_PER_KM,
    DISTRIBUTIONS,
    N_VEHICLES,
    PENETRATION_RATES,
    PCRUISE,
    IDMParameters,
    MEASUREMENT_DISTANCE_KM,
    SimulationConfig,
    VTMicroFuelModel,
    WARMUP_DISTANCE_KM,
    idm_acceleration,
)


PERTURBATION_BRAKE_S = 0.6
PERTURBATION_AFTER_WARMUP_KM = 0.5
COMFORT_WINDOW_BEFORE_S = 5.0
COMFORT_WINDOW_AFTER_S = 30.0


def assign_line_vehicles(
    n_vehicles: int,
    penetration: float,
    distribution: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign ACC cars along a line; clusters do not wrap around."""
    n_acc = int(round(n_vehicles * penetration))
    mask = np.zeros(n_vehicles, dtype=bool)
    if n_acc == 0:
        return mask
    if n_acc == n_vehicles:
        return np.ones(n_vehicles, dtype=bool)

    if distribution == "equal":
        phase = rng.random()
        indices = np.floor((np.arange(n_acc) + phase) * n_vehicles / n_acc).astype(int)
        mask[indices] = True
    elif distribution == "unequal":
        start = int(rng.integers(0, n_vehicles - n_acc + 1))
        mask[start : start + n_acc] = True
    else:
        raise ValueError("distribution must be 'equal' or 'unequal'.")
    return mask


def idm_equilibrium_speed(net_gap_m: float, parameters: IDMParameters) -> float:
    """Solve the zero-relative-speed IDM equilibrium for the initial condition."""
    low, high = 0.0, parameters.desired_speed
    for _ in range(60):
        speed = 0.5 * (low + high)
        free_term = (speed / parameters.desired_speed) ** parameters.exponent
        interaction_term = (
            (parameters.minimum_gap + speed * parameters.time_headway) / net_gap_m
        ) ** 2
        if free_term + interaction_term > 1.0:
            high = speed
        else:
            low = speed
    return 0.5 * (low + high)


def simulate_open_platoon(
    *,
    config: SimulationConfig,
    density_veh_per_km: float,
    penetration: float,
    distribution: str,
    acc_parameters: Optional[ACCParameters],
    fuel_model: VTMicroFuelModel,
    perturb: bool,
    retain_trace: bool = False,
) -> dict:
    """Simulate a finite platoon following a virtual lead on a straight road."""
    if density_veh_per_km <= 0.0:
        raise ValueError("Traffic density must be positive.")
    if config.warmup_distance_km < 0.0 or config.measurement_distance_km <= 0.0:
        raise ValueError("Warm-up distance must be nonnegative and measurement distance positive.")
    rng = np.random.default_rng(config.seed)
    n = config.n_vehicles
    spacing = 1000.0 / density_veh_per_km
    net_gap = spacing - config.vehicle_length
    if net_gap <= 0.0:
        raise ValueError("Density implies overlapping vehicles; reduce density.")

    acc_mask = (
        assign_line_vehicles(n, penetration, distribution, rng)
        if acc_parameters is not None
        else np.zeros(n, dtype=bool)
    )
    acc_indices = np.flatnonzero(acc_mask)
    idm = IDMParameters()
    initial_speed = idm_equilibrium_speed(net_gap, idm)

    # Indices run from the rear of the platoon to its front.
    positions = (np.arange(n, dtype=float) - (n - 1)) * spacing
    speeds = np.full(n, initial_speed, dtype=float)
    virtual_lead_position = positions[-1] + spacing
    virtual_lead_speed = initial_speed

    step_count = int(np.ceil(config.duration / config.dt))
    warmup_distance_m = config.warmup_distance_km * 1000.0
    measurement_end_m = warmup_distance_m + config.measurement_distance_km * 1000.0
    vehicle_distance_m = np.zeros(n, dtype=float)
    fuel_by_vehicle_ml = np.zeros(n, dtype=float)
    measured_distance_by_vehicle_m = np.zeros(n, dtype=float)
    measured_time_by_vehicle_s = np.zeros(n, dtype=float)
    acceleration_abs_time_integral = 0.0
    jerk_squared_sum = 0.0
    comfort_jerk_squared_sum = 0.0
    comfort_acceleration_squared_sum = 0.0
    comfort_vehicle_seconds = 0.0
    minimum_gap_after_warmup = float("inf")
    previous_acceleration = np.zeros(n, dtype=float)
    speed_trace = np.empty((step_count, n), dtype=np.float32) if retain_trace else None
    position_trace = np.empty((step_count, n), dtype=np.float32) if retain_trace else None
    lead_position_trace = np.empty(step_count, dtype=np.float32) if retain_trace else None
    initial_virtual_lead_position = virtual_lead_position
    perturbation_start_s = None
    completed_steps = 0

    for step in range(step_count):
        time_s = step * config.dt
        leader_positions = np.concatenate((positions[1:], [virtual_lead_position]))
        leader_speeds = np.concatenate((speeds[1:], [virtual_lead_speed]))
        gaps = np.maximum(leader_positions - positions - config.vehicle_length, 0.1)

        acceleration = idm_acceleration(gaps, speeds, leader_speeds, idm)
        if acc_parameters is not None and acc_indices.size:
            desired_gaps = acc_parameters.g_min + acc_parameters.tau * speeds[acc_indices]
            spacing_errors = gaps[acc_indices] - desired_gaps
            relative_speeds = leader_speeds[acc_indices] - speeds[acc_indices]
            commands = (
                acc_parameters.alpha * spacing_errors
                + acc_parameters.lam * relative_speeds
            )
            acceleration[acc_indices] = np.clip(
                commands, acc_parameters.a_min, acc_parameters.a_max
            )

        if (
            perturb
            and perturbation_start_s is None
            and virtual_lead_position - initial_virtual_lead_position
            >= (config.warmup_distance_km + PERTURBATION_AFTER_WARMUP_KM) * 1000.0
        ):
            perturbation_start_s = time_s

        # The virtual lead brakes and then returns to its original speed.
        lead_acceleration = 0.0
        if (
            perturbation_start_s is not None
            and perturbation_start_s
            <= time_s
            < perturbation_start_s + PERTURBATION_BRAKE_S
        ):
            lead_acceleration = -2.5
        elif (
            perturbation_start_s is not None
            and perturbation_start_s + PERTURBATION_BRAKE_S
            <= time_s
            < perturbation_start_s + 2.0 * PERTURBATION_BRAKE_S
        ):
            lead_acceleration = 2.5

        next_speeds = np.maximum(0.0, speeds + acceleration * config.dt)
        step_distance = 0.5 * (speeds + next_speeds) * config.dt
        next_positions = positions + step_distance
        next_lead_speed = max(0.0, virtual_lead_speed + lead_acceleration * config.dt)
        next_lead_position = virtual_lead_position + 0.5 * (
            virtual_lead_speed + next_lead_speed
        ) * config.dt

        jerk = (acceleration - previous_acceleration) / config.dt
        next_vehicle_distance_m = vehicle_distance_m + step_distance
        measured_distance = np.maximum(
            0.0,
            np.minimum(next_vehicle_distance_m, measurement_end_m)
            - np.maximum(vehicle_distance_m, warmup_distance_m),
        )
        measurement_fraction = np.divide(
            measured_distance,
            step_distance,
            out=np.zeros_like(measured_distance),
            where=step_distance > 0.0,
        )
        measured_time = config.dt * measurement_fraction
        if (
            perturbation_start_s is not None
            and perturbation_start_s - COMFORT_WINDOW_BEFORE_S
            <= time_s
            < perturbation_start_s
            + 2.0 * PERTURBATION_BRAKE_S
            + COMFORT_WINDOW_AFTER_S
        ):
            controller_time = measured_time[acc_mask]
            comfort_acceleration_squared_sum += float(
                np.sum(acceleration[acc_mask] ** 2 * controller_time)
            )
            comfort_jerk_squared_sum += float(
                np.sum(jerk[acc_mask] ** 2 * controller_time)
            )
            comfort_vehicle_seconds += float(np.sum(controller_time))

        rate = fuel_model.rate_ml_per_s(next_speeds, acceleration)
        fuel_by_vehicle_ml += rate * measured_time
        measured_distance_by_vehicle_m += measured_distance
        measured_time_by_vehicle_s += measured_time
        acceleration_abs_time_integral += float(
            np.sum(np.abs(acceleration) * measured_time)
        )
        jerk_squared_sum += float(np.sum(jerk**2 * measured_time))
        if np.any(measured_distance > 0.0):
            active_vehicles = measured_distance > 0.0
            minimum_gap_after_warmup = min(
                minimum_gap_after_warmup,
                float(np.min(gaps[active_vehicles])),
            )

        positions, speeds = next_positions, next_speeds
        virtual_lead_position, virtual_lead_speed = next_lead_position, next_lead_speed
        vehicle_distance_m = next_vehicle_distance_m
        previous_acceleration = acceleration.copy()
        completed_steps = step + 1
        if retain_trace:
            speed_trace[step] = speeds
            position_trace[step] = positions
            lead_position_trace[step] = virtual_lead_position
        if np.all(vehicle_distance_m >= measurement_end_m):
            break
    else:
        completed_count = int(np.count_nonzero(vehicle_distance_m >= measurement_end_m))
        raise RuntimeError(
            "The platoon did not complete the configured distance window within "
            f"{config.duration:g} simulated seconds ({completed_count}/{n} vehicles completed)."
        )

    total_vehicle_km = float(np.sum(measured_distance_by_vehicle_m) / 1000.0)
    measured_seconds = float(np.sum(measured_time_by_vehicle_s))
    mean_speed_mps = total_vehicle_km * 1000.0 / measured_seconds if measured_seconds else 0.0
    time_trace = (
        np.arange(completed_steps, dtype=float) * config.dt
        if retain_trace
        else None
    )
    return {
        "density": density_veh_per_km,
        "flow": density_veh_per_km * mean_speed_mps * 3.6,
        "mean_speed_mps": mean_speed_mps,
        "mean_abs_acceleration_mps2": acceleration_abs_time_integral / measured_seconds
        if measured_seconds
        else 0.0,
        "rms_jerk_mps3": np.sqrt(jerk_squared_sum / measured_seconds)
        if measured_seconds
        else 0.0,
        "acc_vehicle_rms_jerk_mps3": np.sqrt(
            comfort_jerk_squared_sum / comfort_vehicle_seconds
        )
        if comfort_vehicle_seconds
        else float("nan"),
        "acc_vehicle_rms_acceleration_mps2": np.sqrt(
            comfort_acceleration_squared_sum / comfort_vehicle_seconds
        )
        if comfort_vehicle_seconds
        else float("nan"),
        "fuel_ml": float(np.sum(fuel_by_vehicle_ml)),
        "fuel_ml_per_vehicle_km": float(np.sum(fuel_by_vehicle_ml)) / total_vehicle_km
        if total_vehicle_km
        else float("nan"),
        "total_vehicle_km": total_vehicle_km,
        "warmup_distance_km_per_vehicle": config.warmup_distance_km,
        "measurement_distance_km_per_vehicle": config.measurement_distance_km,
        "simulation_duration_s": completed_steps * config.dt,
        "minimum_gap_m": minimum_gap_after_warmup,
        "fuel_by_vehicle_ml": fuel_by_vehicle_ml,
        "perturbation_start_s": perturbation_start_s,
        "initial_speed_mps": initial_speed,
        "time_s": time_trace,
        "speed_trace_mps": speed_trace[:completed_steps] if retain_trace else None,
        "position_trace_m": position_trace[:completed_steps] if retain_trace else None,
        "lead_position_trace_m": lead_position_trace[:completed_steps] if retain_trace else None,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty results table.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary = []
    for group_key, group_rows in groups.items():
        row = dict(zip(keys, group_key))
        row["n_runs"] = len(group_rows)
        for metric in metrics:
            values = np.asarray([item[metric] for item in group_rows], dtype=float)
            values = values[np.isfinite(values)]
            row["mean_" + metric] = float(np.mean(values)) if values.size else float("nan")
            row["ci95_" + metric] = (
                float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0 if len(values) == 1 else float("nan")
            )
        summary.append(row)
    return summary


def _summary_fields(keys: list[str], metrics: list[str]) -> list[str]:
    fields = list(keys)
    for metric in metrics:
        fields.extend(("mean_" + metric, "ci95_" + metric))
    fields.append("n_runs")
    return fields


def _write_platoon_visualization(
    config: SimulationConfig,
    fuel_model: VTMicroFuelModel,
    output_file: Path,
) -> None:
    """Write a self-contained browser animation of matched 50% ACC platoons."""
    scenarios = []
    frame_stride = max(1, int(round(0.2 / config.dt)))
    for controller in ACC_VARIANTS:
        result = simulate_open_platoon(
            config=config,
            density_veh_per_km=30.0,
            penetration=0.5,
            distribution="equal",
            acc_parameters=controller,
            fuel_model=fuel_model,
            perturb=True,
            retain_trace=True,
        )
        acc_mask = assign_line_vehicles(
            config.n_vehicles,
            0.5,
            "equal",
            np.random.default_rng(config.seed),
        )
        scenarios.append((controller, result, acc_mask))

    frame_dt = config.dt * frame_stride
    common_end_s = min(result["time_s"][-1] for _, result, _ in scenarios)
    frame_times = np.arange(int(np.floor(common_end_s / frame_dt)) + 1) * frame_dt
    tracks = []
    for controller, result, acc_mask in scenarios:
        positions = np.column_stack(
            [
                np.interp(
                    frame_times,
                    result["time_s"],
                    result["position_trace_m"][:, vehicle],
                )
                for vehicle in range(config.n_vehicles)
            ]
        )
        speeds = np.column_stack(
            [
                np.interp(
                    frame_times,
                    result["time_s"],
                    result["speed_trace_mps"][:, vehicle],
                )
                for vehicle in range(config.n_vehicles)
            ]
        )
        lead_positions = np.interp(
            frame_times,
            result["time_s"],
            result["lead_position_trace_m"],
        )
        tracks.append(
            {
                "name": controller.name,
                "types": ["ACC" if is_acc else "Human" for is_acc in acc_mask],
                "positions": np.round(positions, 1).tolist(),
                "meanSpeeds": np.round(np.mean(speeds, axis=1), 2).tolist(),
                "leadPositions": np.round(lead_positions, 1).tolist(),
            }
        )

    data = {
        "dt": frame_dt,
        "times": np.round(frame_times, 2).tolist(),
        "density": 30.0,
        "penetration": 50,
        "tracks": tracks,
        "vehicleLength": config.vehicle_length,
    }
    html = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Straight-road platoon simulation</title>
<style>
  :root { color-scheme: light; font-family: system-ui, Segoe UI, sans-serif; }
  body { max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #17212b; }
  h1 { margin-bottom: 4px; } .note { color: #4b5563; margin-top: 4px; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 18px 0; }
  button, select { font: inherit; padding: 7px 12px; }
  input[type=range] { width: min(620px, 70vw); }
  canvas { display: block; width: 100%; border: 1px solid #cbd5e1; border-radius: 8px; }
  #road { background: #e8eef2; } #map { margin-top: 12px; background: #f8fafc; }
  .legend { display: flex; gap: 18px; margin: 12px 0; }
  .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 5px; vertical-align: -2px; }
  #status { font-variant-numeric: tabular-nums; min-width: 240px; }
</style>
</head>
<body>
<h1>Straight-road vehicle platoon</h1>
<p class="note">100 vehicles, 30 vehicles/km, 50% equally interleaved ACC, one braking pulse.
Choose a controller to compare under the same vehicle placement and initial conditions.
The view follows the platoon midpoint; the strip below shows all vehicles.</p>
<div class="controls">
  <label>Controller <select id="scenario"></select></label>
  <button id="play">Play</button>
  <button id="reset">Reset</button>
  <label>Playback <select id="rate"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
  <span id="status"></span>
</div>
<input id="frame" type="range" min="0" value="0" step="1" aria-label="Simulation time">
<canvas id="road" height="250" aria-label="Animated close view of the platoon"></canvas>
<div class="legend">
  <span><i class="swatch" style="background:#64748b"></i>Human / IDM</span>
  <span><i class="swatch" style="background:#138a72"></i>ACC vehicle</span>
  <span><i class="swatch" style="background:#db7c21"></i>Virtual lead</span>
</div>
<canvas id="map" height="76" aria-label="All vehicles in the platoon"></canvas>
<p class="note">Visualization is a recorded simulation, not a live traffic feed. Car lengths
are enlarged for visibility. Longitudinal controller acceleration is applied directly;
actuator lag and lateral motion are not modeled.</p>
<script>
const data = __SIMULATION_DATA__;
const colors = { Human: "#64748b", ACC: "#138a72" };
const selector = document.getElementById("scenario");
const slider = document.getElementById("frame");
const playButton = document.getElementById("play");
const rateSelector = document.getElementById("rate");
const status = document.getElementById("status");
const road = document.getElementById("road");
const map = document.getElementById("map");
const roadCtx = road.getContext("2d");
const mapCtx = map.getContext("2d");
data.tracks.forEach((track, i) => {
  const option = document.createElement("option");
  option.value = i; option.textContent = track.name; selector.append(option);
});
slider.max = String(data.times.length - 1);
let frame = 0, playing = false, previousTime = 0, fractionalFrame = 0;
const fitCanvas = (canvas, ctx) => {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Number(canvas.getAttribute("height"));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width, height };
};
function draw() {
  const track = data.tracks[Number(selector.value)];
  const { width, height } = fitCanvas(road, roadCtx);
  const t = data.times[frame], positions = track.positions[frame];
  const center = positions[Math.floor(positions.length / 2)];
  const worldWidth = 1200, scale = width / worldWidth;
  const xAt = x => width / 2 + (x - center) * scale;
  roadCtx.clearRect(0, 0, width, height);
  roadCtx.fillStyle = "#dce5e9"; roadCtx.fillRect(0, height * .28, width, height * .48);
  roadCtx.fillStyle = "#fff"; roadCtx.fillRect(0, height * .51, width, 3);
  roadCtx.fillStyle = "#91a0aa"; roadCtx.font = "13px system-ui";
  roadCtx.fillText("Rear", 12, 24); roadCtx.fillText("Travel direction →", width - 150, 24);
  for (let i = 0; i < positions.length; i++) {
    const x = xAt(positions[i]), carWidth = Math.max(14, data.vehicleLength * scale);
    if (x < -carWidth || x > width + carWidth) continue;
    roadCtx.fillStyle = colors[track.types[i]];
    roadCtx.beginPath();
    roadCtx.roundRect(x - carWidth / 2, height * .38, carWidth, height * .25, 4);
    roadCtx.fill();
    roadCtx.fillStyle = "#fff"; roadCtx.font = "10px system-ui";
    if (carWidth > 22) roadCtx.fillText(String(i + 1), x - carWidth / 2 + 3, height * .55);
  }
  const leadX = xAt(track.leadPositions[frame]);
  if (leadX > -20 && leadX < width + 20) {
    roadCtx.fillStyle = "#db7c21"; roadCtx.fillRect(leadX - 10, height * .36, 20, height * .29);
    roadCtx.fillStyle = "#17212b"; roadCtx.fillText("Lead", leadX - 15, height * .32);
  }
  const mapSize = fitCanvas(map, mapCtx);
  mapCtx.clearRect(0, 0, mapSize.width, mapSize.height);
  mapCtx.fillStyle = "#e8eef2"; mapCtx.fillRect(0, 25, mapSize.width, 25);
  const lo = Math.min(...positions), hi = Math.max(...positions), span = Math.max(1, hi - lo);
  positions.forEach((x, i) => {
    mapCtx.fillStyle = colors[track.types[i]];
    mapCtx.beginPath();
    mapCtx.arc(8 + (x - lo) / span * (mapSize.width - 16), 37, 3, 0, 2 * Math.PI);
    mapCtx.fill();
  });
  mapCtx.fillStyle = "#334155"; mapCtx.font = "12px system-ui";
  mapCtx.fillText("All 100 vehicles — rear", 8, 17);
  mapCtx.fillText("front", mapSize.width - 42, 17);
  const meanSpeed = track.meanSpeeds[frame] * 3.6;
  status.textContent = `t = ${t.toFixed(1)} s · mean speed ${meanSpeed.toFixed(1)} km/h`;
  slider.value = String(frame);
}
function animate(now) {
  if (playing) {
    if (previousTime) fractionalFrame += (now - previousTime) / 1000 * Number(rateSelector.value) / data.dt;
    frame = Math.min(data.times.length - 1, Math.floor(fractionalFrame));
    if (frame >= data.times.length - 1) { playing = false; playButton.textContent = "Play"; }
  }
  previousTime = now;
  draw();
  requestAnimationFrame(animate);
}
playButton.addEventListener("click", () => {
  if (frame >= data.times.length - 1) frame = 0;
  fractionalFrame = frame; playing = !playing; previousTime = 0;
  playButton.textContent = playing ? "Pause" : "Play";
});
document.getElementById("reset").addEventListener("click", () => {
  playing = false; frame = 0; fractionalFrame = 0; previousTime = 0; playButton.textContent = "Play"; draw();
});
slider.addEventListener("input", () => { frame = Number(slider.value); fractionalFrame = frame; draw(); });
selector.addEventListener("change", draw);
window.addEventListener("resize", draw);
requestAnimationFrame(animate);
</script>
</body>
</html>"""
    output_file.write_text(
        html.replace("__SIMULATION_DATA__", json.dumps(data, separators=(",", ":"))),
        encoding="utf-8",
    )


def run_experiments(
    config: SimulationConfig,
    fuel_model: VTMicroFuelModel,
    output_dir: Path,
    replicates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_rows = []
    fd_runs = []
    fd_controllers = ((None, "human-only"),) + tuple((controller, controller.name) for controller in ACC_VARIANTS)

    for density in DENSITIES_VEH_PER_KM:
        for controller, label in fd_controllers:
            for replicate in range(replicates):
                seed = config.seed + replicate
                result = simulate_open_platoon(
                    config=replace(config, seed=seed),
                    density_veh_per_km=density,
                    penetration=0.0 if controller is None else 0.5,
                    distribution="equal",
                    acc_parameters=controller,
                    fuel_model=fuel_model,
                    perturb=True,
                )
                fd_runs.append(
                    {
                        "controller": label,
                        "density_veh_per_km": density,
                        "replicate": replicate,
                        "seed": seed,
                        "flow_veh_per_hour": result["flow"],
                        "mean_speed_mps": result["mean_speed_mps"],
                        "fuel_ml": result["fuel_ml"],
                        "fuel_ml_per_vehicle_km": result["fuel_ml_per_vehicle_km"],
                        "total_vehicle_km": result["total_vehicle_km"],
                        "warmup_distance_km_per_vehicle": result[
                            "warmup_distance_km_per_vehicle"
                        ],
                        "measurement_distance_km_per_vehicle": result[
                            "measurement_distance_km_per_vehicle"
                        ],
                        "simulation_duration_s": result["simulation_duration_s"],
                    }
                )
                vehicle_rows.extend(
                    {
                        "scenario": "fundamental_diagram",
                        "controller": label,
                        "density_veh_per_km": density,
                        "penetration": 0.0 if controller is None else 0.5,
                        "distribution": "equal",
                        "replicate": replicate,
                        "seed": seed,
                        "vehicle_index": vehicle_index,
                        "fuel_ml_distance_window": float(fuel_ml),
                    }
                    for vehicle_index, fuel_ml in enumerate(result["fuel_by_vehicle_ml"])
                )

    fd_keys = ["controller", "density_veh_per_km"]
    fd_metrics = [
        "flow_veh_per_hour",
        "mean_speed_mps",
        "fuel_ml_per_vehicle_km",
        "simulation_duration_s",
    ]
    fd_summary = _summarize(fd_runs, fd_keys, fd_metrics)
    _write_csv(output_dir / "fundamental_diagram_runs.csv", fd_runs)
    _write_csv(output_dir / "fundamental_diagram.csv", fd_summary)

    fuel_runs = []
    for controller in ACC_VARIANTS:
        for penetration in PENETRATION_RATES:
            for distribution in DISTRIBUTIONS:
                for replicate in range(replicates):
                    seed = config.seed + replicate
                    result = simulate_open_platoon(
                        config=replace(config, seed=seed),
                        density_veh_per_km=30.0,
                        penetration=penetration,
                        distribution=distribution,
                        acc_parameters=controller,
                        fuel_model=fuel_model,
                        perturb=True,
                    )
                    fuel_runs.append(
                        {
                            "controller": controller.name,
                            "penetration": penetration,
                            "distribution": distribution,
                            "density_veh_per_km": 30.0,
                            "replicate": replicate,
                            "seed": seed,
                            "fuel_ml": result["fuel_ml"],
                            "fuel_ml_per_vehicle_km": result["fuel_ml_per_vehicle_km"],
                            "mean_speed_mps": result["mean_speed_mps"],
                            "total_vehicle_km": result["total_vehicle_km"],
                            "minimum_gap_m": result["minimum_gap_m"],
                            "warmup_distance_km_per_vehicle": result[
                                "warmup_distance_km_per_vehicle"
                            ],
                            "measurement_distance_km_per_vehicle": result[
                                "measurement_distance_km_per_vehicle"
                            ],
                            "simulation_duration_s": result["simulation_duration_s"],
                            "acc_vehicle_rms_jerk_mps3": result[
                                "acc_vehicle_rms_jerk_mps3"
                            ],
                            "acc_vehicle_rms_acceleration_mps2": result[
                                "acc_vehicle_rms_acceleration_mps2"
                            ],
                        }
                    )
                    vehicle_rows.extend(
                        {
                            "scenario": "fleet_fuel_usage",
                            "controller": controller.name,
                            "density_veh_per_km": 30.0,
                            "penetration": penetration,
                            "distribution": distribution,
                            "replicate": replicate,
                            "seed": seed,
                            "vehicle_index": vehicle_index,
                            "fuel_ml_distance_window": float(fuel_ml),
                        }
                        for vehicle_index, fuel_ml in enumerate(result["fuel_by_vehicle_ml"])
                    )

    fuel_keys = ["controller", "penetration", "distribution", "density_veh_per_km"]
    fuel_metrics = [
        "fuel_ml",
        "fuel_ml_per_vehicle_km",
        "mean_speed_mps",
        "total_vehicle_km",
        "minimum_gap_m",
        "simulation_duration_s",
        "acc_vehicle_rms_jerk_mps3",
        "acc_vehicle_rms_acceleration_mps2",
    ]
    fuel_summary = _summarize(fuel_runs, fuel_keys, fuel_metrics)
    _write_csv(output_dir / "fleet_fuel_usage_runs.csv", fuel_runs)
    _write_csv(output_dir / "fleet_fuel_usage.csv", fuel_summary)
    _write_csv(output_dir / "vehicle_fuel_burn.csv", vehicle_rows)

    # Compare pulse and no-pulse traces for the same open-road platoon.
    baseline = simulate_open_platoon(
        config=config,
        density_veh_per_km=30.0,
        penetration=0.5,
        distribution="equal",
        acc_parameters=PCRUISE,
        fuel_model=fuel_model,
        perturb=False,
        retain_trace=True,
    )
    perturbed = simulate_open_platoon(
        config=config,
        density_veh_per_km=30.0,
        penetration=0.5,
        distribution="equal",
        acc_parameters=PCRUISE,
        fuel_model=fuel_model,
        perturb=True,
        retain_trace=True,
    )
    common_steps = min(len(perturbed["time_s"]), len(baseline["time_s"]))
    deviations = (
        perturbed["speed_trace_mps"][:common_steps]
        - baseline["speed_trace_mps"][:common_steps]
    )
    rms = np.sqrt(np.mean(deviations.astype(float) ** 2, axis=1))
    maximum = np.max(np.abs(deviations), axis=1)
    times = perturbed["time_s"][:common_steps]
    pulse_start_s = perturbed["perturbation_start_s"]
    pulse_end_s = pulse_start_s + 2.0 * PERTURBATION_BRAKE_S

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    vmax = max(0.1, float(np.max(np.abs(deviations))))
    image = axes[0].imshow(
        deviations.T,
        aspect="auto",
        origin="lower",
        extent=(times[0], times[-1], 0, config.n_vehicles - 1),
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0].axvspan(
        pulse_start_s,
        pulse_end_s,
        color="black",
        alpha=0.15,
        label="virtual-lead braking pulse",
    )
    axes[0].set(
        title="Straight-road platoon pulse response (common elapsed-time prefix)",
        ylabel="Vehicle index (rear to front)",
    )
    axes[0].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0], label="Speed difference from no-pulse run (m/s)")
    axes[1].plot(times, rms, label="Fleet RMS response")
    axes[1].plot(times, maximum, label="Maximum absolute response", alpha=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="Pulse response (m/s)")
    axes[1].legend()
    fig.savefig(output_dir / "stability_perturbation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for _, label in fd_controllers:
        selected = sorted(
            (row for row in fd_summary if row["controller"] == label),
            key=lambda row: row["density_veh_per_km"],
        )
        ax.errorbar(
            [row["density_veh_per_km"] for row in selected],
            [row["mean_flow_veh_per_hour"] for row in selected],
            yerr=[row["ci95_flow_veh_per_hour"] for row in selected],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set(
        title="Straight-road kinematic flow estimate (mean ± 95% CI)",
        xlabel="Initial platoon density (vehicles/km)",
        ylabel="Estimated flow (vehicles/hour)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "fundamental_diagram.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        3,
        len(ACC_VARIANTS),
        figsize=(15, 12),
        sharex=True,
        constrained_layout=True,
    )
    for column, controller in enumerate(ACC_VARIANTS):
        for distribution in DISTRIBUTIONS:
            selected = sorted(
                (
                    row
                    for row in fuel_summary
                    if row["controller"] == controller.name and row["distribution"] == distribution
                ),
                key=lambda row: row["penetration"],
            )
            x = [100.0 * row["penetration"] for row in selected]
            axes[0, column].errorbar(
                x,
                [row["mean_fuel_ml_per_vehicle_km"] for row in selected],
                yerr=[row["ci95_fuel_ml_per_vehicle_km"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
            axes[1, column].errorbar(
                x,
                [row["mean_fuel_ml"] for row in selected],
                yerr=[row["ci95_fuel_ml"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
            axes[2, column].errorbar(
                x,
                [row["mean_mean_speed_mps"] * 3.6 for row in selected],
                yerr=[row["ci95_mean_speed_mps"] * 3.6 for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
        axes[0, column].set_title(controller.name)
        for row_index in range(3):
            axes[row_index, column].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Fuel\n(mL / vehicle-km)")
    axes[1, 0].set_ylabel(
        f"Fleet fuel\n(mL per {config.measurement_distance_km:g} km/vehicle)"
    )
    axes[2, 0].set_ylabel("Mean speed\n(km/h)")
    for ax in axes[-1, :]:
        ax.set_xlabel("ACC penetration (%)")
    axes[0, -1].legend(title="Placement")
    fig.suptitle(
        "Straight-road fuel and speed at 30 vehicles/km "
        f"({config.measurement_distance_km:g} km/vehicle after "
        f"{config.warmup_distance_km:g} km warm-up)"
    )
    fig.savefig(output_dir / "fleet_fuel_usage.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        len(ACC_VARIANTS),
        figsize=(15, 9),
        constrained_layout=True,
    )
    for column, controller in enumerate(ACC_VARIANTS):
        for distribution in DISTRIBUTIONS:
            selected = sorted(
                (
                    row
                    for row in fuel_summary
                    if row["controller"] == controller.name
                    and row["distribution"] == distribution
                    and row["penetration"] > 0.0
                ),
                key=lambda row: row["penetration"],
            )
            axes[0, column].errorbar(
                [100.0 * row["penetration"] for row in selected],
                [row["mean_acc_vehicle_rms_jerk_mps3"] for row in selected],
                yerr=[row["ci95_acc_vehicle_rms_jerk_mps3"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
            axes[1, column].errorbar(
                [100.0 * row["penetration"] for row in selected],
                [row["mean_acc_vehicle_rms_acceleration_mps2"] for row in selected],
                yerr=[row["ci95_acc_vehicle_rms_acceleration_mps2"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
        axes[0, column].set_title(controller.name)
        axes[1, column].set_xlabel("ACC penetration (%)")
        for row_index in range(2):
            axes[row_index, column].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("ACC-car RMS jerk (m/s^3)")
    axes[1, 0].set_ylabel("ACC-car RMS acceleration (m/s^2)")
    axes[0, -1].legend(title="Placement")
    fig.suptitle(
        "ACC-vehicle longitudinal comfort proxies (mean +/- 95% CI) "
        f"within {config.measurement_distance_km:g} measured km per vehicle"
    )
    fig.savefig(output_dir / "ride_smoothness.png", dpi=160)
    plt.close(fig)

    visualization_file = output_dir / "platoon_animation.html"
    _write_platoon_visualization(config, fuel_model, visualization_file)
    print("Saved straight-road results to:", output_dir.resolve())
    print("The ride-comfort figure covers ACC vehicles during the braking-pulse response.")
    print("Jerk uses finite differences of commanded acceleration; actuator lag is not modeled.")
    print(
        f"Each vehicle contributes {config.measurement_distance_km:g} measured km "
        "after the distance-based warm-up."
    )
    print("The 95% intervals reflect seeded placement variation only.")
    print("Open the platoon animation at:", visualization_file.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=1800.0,
        help="Maximum simulated time allowed to complete each distance window (seconds).",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step in seconds.")
    parser.add_argument(
        "--distance-km",
        type=float,
        default=MEASUREMENT_DISTANCE_KM,
        help="Measured travel distance per vehicle, after warm-up.",
    )
    parser.add_argument(
        "--warmup-distance-km",
        type=float,
        default=WARMUP_DISTANCE_KM,
        help="Unmeasured distance per vehicle for startup transients.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--replicates", type=int, default=5, help="Seeded runs per scenario.")
    parser.add_argument(
        "--open-visualization",
        action="store_true",
        help="Open the generated platoon animation in the default browser.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "straight_road_simulation",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0 or args.duration <= 0.0:
        raise ValueError("dt and maximum simulated duration must be positive.")
    if args.distance_km <= 0.0 or args.warmup_distance_km < 0.0:
        raise ValueError("Measurement distance must be positive and warm-up distance nonnegative.")
    if args.replicates < 1:
        raise ValueError("replicates must be at least one.")
    config = SimulationConfig(
        dt=args.dt,
        duration=args.duration,
        seed=args.seed,
        warmup_distance_km=args.warmup_distance_km,
        measurement_distance_km=args.distance_km,
    )
    fuel_model = VTMicroFuelModel.provided_default()
    run_experiments(config, fuel_model, args.output_dir, args.replicates)
    if args.open_visualization:
        webbrowser.open((args.output_dir / "platoon_animation.html").resolve().as_uri())


if __name__ == "__main__":
    main()
