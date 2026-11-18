#!/usr/bin/env python3
"""Microscopic mixed-traffic simulation on a single-lane circular road.

Run with:
    python traffic_ring_simulation.py

The default VT-Micro matrix is the coefficient supplied for this project and
is used for both acceleration signs. It operates on speed in km/h and signed
acceleration in km/h/s; each matrix entry K[i, j] multiplies speed**i *
acceleration**j. Supply separate matrices in JSON with --fuel-coefficients if
you have distinct positive- and negative-acceleration calibrations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


N_VEHICLES = 100
PENETRATION_RATES = (0.0, 0.25, 0.50, 0.75, 1.0)
DISTRIBUTIONS = ("equal", "unequal")
DENSITIES_VEH_PER_KM = (10.0, 20.0, 30.0, 40.0, 50.0)
CAR_LENGTH_M = 4.5
WARMUP_DISTANCE_KM = 0.5
MEASUREMENT_DISTANCE_KM = 3.0
PERTURBATION_DISTANCE_AFTER_WARMUP_KM = 0.5
PERTURBATION_DURATION_S = 0.6


@dataclass(frozen=True)
class ACCParameters:
    """Parameters for the specified proportional-derivative ACC law."""

    name: str
    alpha: float
    tau: float
    g_min: float = 2.0
    lam: float = 0.05
    a_min: float = -5.0
    a_max: float = 4.0


PCRUISE = ACCParameters("pcruise", alpha=0.8, tau=2.5)
AGGRESSIVE_ACC = ACCParameters("aggressive ACC", alpha=1.2, tau=1.5)
CONSERVATIVE_ACC = ACCParameters("conservative ACC", alpha=0.5, tau=3.5)
ACC_VARIANTS = (PCRUISE, AGGRESSIVE_ACC, CONSERVATIVE_ACC)


@dataclass(frozen=True)
class IDMParameters:
    """Standard Intelligent Driver Model parameters for human drivers."""

    desired_speed: float = 30.0
    time_headway: float = 1.4
    minimum_gap: float = 2.0
    maximum_acceleration: float = 1.4
    comfortable_deceleration: float = 2.0
    exponent: int = 4


@dataclass(frozen=True)
class SimulationConfig:
    n_vehicles: int = N_VEHICLES
    dt: float = 0.1
    duration: float = 1800.0
    vehicle_length: float = CAR_LENGTH_M
    seed: int = 2026
    warmup_distance_km: float = WARMUP_DISTANCE_KM
    measurement_distance_km: float = MEASUREMENT_DISTANCE_KM


# User-provided VT-Micro matrix. The same matrix is applied to both acceleration
# signs, per the selected interpretation. Output is ln(mL/s).
DEFAULT_FUEL_K_POSITIVE = np.array(
    [
        [-3.0, 0.749910, -0.014320, 0.000213],
        [0.082150, -0.005120, 0.000240, -0.000004],
        [-0.001140, 0.000074, -0.000003, 0.000000],
        [0.000006, -0.000000, 0.000000, 0.000000],
    ],
    dtype=float,
)
DEFAULT_FUEL_K_NEGATIVE = np.array(
    [
        [-3.0, 0.749910, -0.014320, 0.000213],
        [0.082150, -0.005120, 0.000240, -0.000004],
        [-0.001140, 0.000074, -0.000003, 0.000000],
        [0.000006, -0.000000, 0.000000, 0.000000],
    ],
    dtype=float,
)


def pcruise_acceleration(
    actual_gap: float,
    v_ego: float,
    v_lead: float,
    parameters: ACCParameters = PCRUISE,
) -> float:
    """Return pcruise's exact PD command after applying acceleration limits."""
    desired_gap = parameters.g_min + parameters.tau * v_ego
    spacing_error = actual_gap - desired_gap
    delta_v = v_lead - v_ego
    command = parameters.alpha * spacing_error + parameters.lam * delta_v
    return float(np.clip(command, parameters.a_min, parameters.a_max))


def idm_acceleration(
    actual_gap: np.ndarray,
    v_ego: np.ndarray,
    v_lead: np.ndarray,
    parameters: IDMParameters,
) -> np.ndarray:
    """Vectorized Intelligent Driver Model acceleration in m/s^2."""
    gap = np.maximum(actual_gap, 0.1)
    delta_v = v_ego - v_lead
    desired_dynamic_gap = parameters.minimum_gap + np.maximum(
        0.0,
        v_ego * parameters.time_headway
        + (v_ego * delta_v)
        / (2.0 * np.sqrt(parameters.maximum_acceleration * parameters.comfortable_deceleration)),
    )
    free_road_term = (v_ego / parameters.desired_speed) ** parameters.exponent
    interaction_term = (desired_dynamic_gap / gap) ** 2
    acceleration = parameters.maximum_acceleration * (1.0 - free_road_term - interaction_term)
    return np.clip(acceleration, -5.0, 4.0)


class VTMicroFuelModel:
    """Piecewise VT-Micro polynomial fuel-rate model.

    VT-Micro uses a separate polynomial for positive and negative acceleration.
    The implementation returns instantaneous fuel rate in mL/s and supports
    loading calibrated coefficient matrices from JSON.
    """

    def __init__(self, k_positive: np.ndarray, k_negative: np.ndarray):
        self.k_positive = np.asarray(k_positive, dtype=float)
        self.k_negative = np.asarray(k_negative, dtype=float)
        if self.k_positive.shape != (4, 4) or self.k_negative.shape != (4, 4):
            raise ValueError("Both VT-Micro matrices must have shape (4, 4).")
        if not np.isfinite(self.k_positive).all() or not np.isfinite(self.k_negative).all():
            raise ValueError("VT-Micro coefficients must all be finite.")

    @classmethod
    def from_json(cls, path: Path) -> "VTMicroFuelModel":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(data["positive_acceleration"], data["negative_acceleration"])

    @classmethod
    def provided_default(cls) -> "VTMicroFuelModel":
        return cls(DEFAULT_FUEL_K_POSITIVE, DEFAULT_FUEL_K_NEGATIVE)

    @staticmethod
    def _polynomial(speed: np.ndarray, acceleration: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        # Polynomial terms are K[i, j] * speed_kph**i * acceleration_kph_per_s**j.
        return np.polynomial.polynomial.polyval2d(speed, acceleration, coefficients)

    def rate_ml_per_s(self, speed_mps: np.ndarray, acceleration_mps2: np.ndarray) -> np.ndarray:
        speed_kph = np.maximum(speed_mps, 0.0) * 3.6
        acceleration_kph_per_s = acceleration_mps2 * 3.6
        positive = acceleration_mps2 >= 0.0
        log_rate = np.empty_like(speed_kph, dtype=float)
        log_rate[positive] = self._polynomial(
            speed_kph[positive],
            acceleration_kph_per_s[positive],
            self.k_positive,
        )
        log_rate[~positive] = self._polynomial(
            speed_kph[~positive],
            acceleration_kph_per_s[~positive],
            self.k_negative,
        )
        # Bound the exponential against numerical overflow while keeping fuel
        # consumption positive.
        return np.exp(np.clip(log_rate, -20.0, 10.0))


def assign_acc_vehicles(
    n_vehicles: int,
    penetration: float,
    distribution: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build an ACC membership mask with even interleaving or one cluster."""
    n_acc = int(round(n_vehicles * penetration))
    mask = np.zeros(n_vehicles, dtype=bool)
    if n_acc == 0:
        return mask
    if n_acc == n_vehicles:
        return np.ones(n_vehicles, dtype=bool)

    if distribution == "equal":
        # A random phase preserves near-equal spacing between ACC vehicles.
        phase = rng.random()
        indices = np.floor((np.arange(n_acc) + phase) * n_vehicles / n_acc).astype(int)
        mask[indices] = True
    elif distribution == "unequal":
        start = int(rng.integers(0, n_vehicles))
        mask[(start + np.arange(n_acc)) % n_vehicles] = True
    else:
        raise ValueError("distribution must be 'equal' or 'unequal'.")
    return mask


def simulate(
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
    """Run one ring-road scenario and return fleet metrics and optional trace."""
    if density_veh_per_km <= 0.0:
        raise ValueError("Traffic density must be positive.")
    if config.warmup_distance_km < 0.0 or config.measurement_distance_km <= 0.0:
        raise ValueError("Warm-up distance must be nonnegative and measurement distance positive.")
    rng = np.random.default_rng(config.seed)
    n = config.n_vehicles
    road_length = n / density_veh_per_km * 1000.0
    spacing = road_length / n
    initial_net_gap = max(0.1, spacing - config.vehicle_length)
    acc_mask = assign_acc_vehicles(n, penetration, distribution, rng) if acc_parameters else np.zeros(n, bool)

    # Start near the pcruise equilibrium speed for this spacing, capped at the
    # human model's free-flow speed. All cars begin at the same speed.
    initial_speed = min(30.0, max(1.0, (initial_net_gap - 2.0) / 2.5))
    positions = np.arange(n, dtype=float) * spacing
    speeds = np.full(n, initial_speed, dtype=float)
    acc_indices = np.flatnonzero(acc_mask)
    idm = IDMParameters()
    step_count = int(np.ceil(config.duration / config.dt))
    warmup_distance_m = config.warmup_distance_km * 1000.0
    measurement_end_m = warmup_distance_m + config.measurement_distance_km * 1000.0
    vehicle_distance_m = np.zeros(n, dtype=float)
    fuel_by_vehicle_ml = np.zeros(n, dtype=float)
    measured_distance_by_vehicle_m = np.zeros(n, dtype=float)
    measured_time_by_vehicle_s = np.zeros(n, dtype=float)
    measured_speed_time_integral = 0.0
    speed_trace = np.empty((step_count, n), dtype=np.float32) if retain_trace else None
    perturbation_start_s = None
    completed_steps = 0

    for step in range(step_count):
        time_s = step * config.dt
        leaders = np.roll(positions, -1)
        gaps = (leaders - positions) % road_length - config.vehicle_length
        gaps = np.maximum(gaps, 0.1)
        lead_speeds = np.roll(speeds, -1)

        acceleration = idm_acceleration(gaps, speeds, lead_speeds, idm)
        if acc_parameters is not None and acc_indices.size:
            desired_gaps = acc_parameters.g_min + acc_parameters.tau * speeds[acc_indices]
            spacing_errors = gaps[acc_indices] - desired_gaps
            relative_speeds = lead_speeds[acc_indices] - speeds[acc_indices]
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
            and vehicle_distance_m[0]
            >= warmup_distance_m + PERTURBATION_DISTANCE_AFTER_WARMUP_KM * 1000.0
        ):
            perturbation_start_s = time_s
        if (
            perturbation_start_s is not None
            and perturbation_start_s
            <= time_s
            < perturbation_start_s + PERTURBATION_DURATION_S
        ):
            acceleration[0] = min(acceleration[0], -2.5)

        next_speeds = np.maximum(0.0, speeds + acceleration * config.dt)
        # Trapezoidal position integration reduces time-step bias.
        step_distance = 0.5 * (speeds + next_speeds) * config.dt
        next_positions = (positions + step_distance) % road_length
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

        rate = fuel_model.rate_ml_per_s(next_speeds, acceleration)
        fuel_by_vehicle_ml += rate * measured_time
        measured_distance_by_vehicle_m += measured_distance
        measured_time_by_vehicle_s += measured_time
        measured_speed_time_integral += float(np.sum(next_speeds * measured_time))

        positions, speeds = next_positions, next_speeds
        vehicle_distance_m = next_vehicle_distance_m
        completed_steps = step + 1
        if retain_trace:
            speed_trace[step] = speeds
        if np.all(vehicle_distance_m >= measurement_end_m):
            break
    else:
        completed_count = int(np.count_nonzero(vehicle_distance_m >= measurement_end_m))
        raise RuntimeError(
            "The fleet did not complete the configured distance window within "
            f"{config.duration:g} simulated seconds ({completed_count}/{n} vehicles completed)."
        )

    total_vehicle_km = float(np.sum(measured_distance_by_vehicle_m) / 1000.0)
    total_measured_time_s = float(np.sum(measured_time_by_vehicle_s))
    mean_speed_mps = (
        measured_speed_time_integral / total_measured_time_s
        if total_measured_time_s > 0.0
        else 0.0
    )
    time_trace = (
        np.arange(completed_steps, dtype=float) * config.dt
        if retain_trace
        else None
    )
    return {
        "density": density_veh_per_km,
        "flow": density_veh_per_km * mean_speed_mps * 3.6,
        "mean_speed_mps": mean_speed_mps,
        "fuel_ml": float(np.sum(fuel_by_vehicle_ml)),
        "fuel_ml_per_vehicle_km": (
            float(np.sum(fuel_by_vehicle_ml)) / total_vehicle_km
            if total_vehicle_km
            else float("nan")
        ),
        "total_vehicle_km": total_vehicle_km,
        "fuel_by_vehicle_ml": fuel_by_vehicle_ml,
        "warmup_distance_km_per_vehicle": config.warmup_distance_km,
        "measurement_distance_km_per_vehicle": config.measurement_distance_km,
        "simulation_duration_s": completed_steps * config.dt,
        "perturbation_start_s": perturbation_start_s,
        "acc_mask": acc_mask,
        "initial_speed_mps": initial_speed,
        "time_s": time_trace,
        "speed_trace_mps": speed_trace[:completed_steps] if retain_trace else None,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_runs(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, group in groups.items():
        summary = dict(zip(group_keys, key))
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            mean = float(np.mean(values))
            ci95 = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            summary["mean_" + metric] = mean
            summary["ci95_" + metric] = ci95
            summary["n_runs"] = len(values)
        summaries.append(summary)
    return summaries


def _summary_fields(group_keys: list[str], metrics: list[str]) -> list[str]:
    fields = list(group_keys)
    for metric in metrics:
        fields.extend(("mean_" + metric, "ci95_" + metric))
    fields.append("n_runs")
    return fields


def run_experiments(
    config: SimulationConfig,
    fuel_model: VTMicroFuelModel,
    output_dir: Path,
    replicates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pair controller comparisons across seeds so random placement is represented.
    fd_runs = []
    vehicle_fuel_rows = []
    fd_controllers = ((None, "human-only"),) + tuple((c, c.name) for c in ACC_VARIANTS)
    for density in DENSITIES_VEH_PER_KM:
        for controller, label in fd_controllers:
            for replicate in range(replicates):
                run_seed = config.seed + replicate
                result = simulate(
                    config=replace(config, seed=run_seed),
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
                        "seed": run_seed,
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
                vehicle_fuel_rows.extend(
                    {
                        "scenario": "fundamental_diagram",
                        "controller": label,
                        "density_veh_per_km": density,
                        "penetration": 0.0 if controller is None else 0.5,
                        "distribution": "equal",
                        "replicate": replicate,
                        "seed": run_seed,
                        "vehicle_index": vehicle_index,
                        "fuel_ml_distance_window": float(fuel_ml),
                    }
                    for vehicle_index, fuel_ml in enumerate(result["fuel_by_vehicle_ml"])
                )
    fd_keys = ["controller", "density_veh_per_km"]
    fd_metrics = [
        "flow_veh_per_hour",
        "mean_speed_mps",
        "fuel_ml",
        "fuel_ml_per_vehicle_km",
        "total_vehicle_km",
        "simulation_duration_s",
    ]
    fd_rows = _summarize_runs(fd_runs, fd_keys, fd_metrics)
    _write_csv(
        output_dir / "fundamental_diagram_runs.csv",
        fd_runs,
        [
            "controller",
            "density_veh_per_km",
            "replicate",
            "seed",
            "flow_veh_per_hour",
            "mean_speed_mps",
            "fuel_ml",
            "fuel_ml_per_vehicle_km",
            "total_vehicle_km",
            "warmup_distance_km_per_vehicle",
            "measurement_distance_km_per_vehicle",
            "simulation_duration_s",
        ],
    )
    _write_csv(
        output_dir / "fundamental_diagram.csv",
        fd_rows,
        _summary_fields(fd_keys, fd_metrics),
    )

    # Fuel comparison at a common density, repeated with paired placement seeds.
    fuel_runs = []
    for controller in ACC_VARIANTS:
        for penetration in PENETRATION_RATES:
            for distribution in DISTRIBUTIONS:
                for replicate in range(replicates):
                    run_seed = config.seed + replicate
                    result = simulate(
                        config=replace(config, seed=run_seed),
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
                            "seed": run_seed,
                            "fuel_ml": result["fuel_ml"],
                            "fuel_ml_per_vehicle_km": result["fuel_ml_per_vehicle_km"],
                            "mean_speed_mps": result["mean_speed_mps"],
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
                    vehicle_fuel_rows.extend(
                        {
                            "scenario": "fleet_fuel_usage",
                            "controller": controller.name,
                            "density_veh_per_km": 30.0,
                            "penetration": penetration,
                            "distribution": distribution,
                            "replicate": replicate,
                            "seed": run_seed,
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
        "simulation_duration_s",
    ]
    fuel_rows = _summarize_runs(fuel_runs, fuel_keys, fuel_metrics)
    _write_csv(
        output_dir / "fleet_fuel_usage_runs.csv",
        fuel_runs,
        [
            "controller",
            "penetration",
            "distribution",
            "density_veh_per_km",
            "replicate",
            "seed",
            "fuel_ml",
            "fuel_ml_per_vehicle_km",
            "mean_speed_mps",
            "total_vehicle_km",
            "warmup_distance_km_per_vehicle",
            "measurement_distance_km_per_vehicle",
            "simulation_duration_s",
        ],
    )
    _write_csv(
        output_dir / "fleet_fuel_usage.csv",
        fuel_rows,
        _summary_fields(fuel_keys, fuel_metrics),
    )
    _write_csv(
        output_dir / "vehicle_fuel_burn.csv",
        vehicle_fuel_rows,
        [
            "scenario",
            "controller",
            "density_veh_per_km",
            "penetration",
            "distribution",
            "replicate",
            "seed",
            "vehicle_index",
            "fuel_ml_distance_window",
        ],
    )

    # Compare against an identical unperturbed run to isolate the braking pulse
    # from normal settling of the mixed human/ACC fleet.
    stability_baseline = simulate(
        config=config,
        density_veh_per_km=30.0,
        penetration=0.5,
        distribution="equal",
        acc_parameters=PCRUISE,
        fuel_model=fuel_model,
        perturb=False,
        retain_trace=True,
    )
    stability = simulate(
        config=config,
        density_veh_per_km=30.0,
        penetration=0.5,
        distribution="equal",
        acc_parameters=PCRUISE,
        fuel_model=fuel_model,
        perturb=True,
        retain_trace=True,
    )
    common_steps = min(len(stability["time_s"]), len(stability_baseline["time_s"]))
    times = stability["time_s"][:common_steps]
    deviations = (
        stability["speed_trace_mps"][:common_steps]
        - stability_baseline["speed_trace_mps"][:common_steps]
    )
    rms_deviation = np.sqrt(np.mean(deviations.astype(float) ** 2, axis=1))
    max_deviation = np.max(np.abs(deviations), axis=1)
    pulse_start_s = stability["perturbation_start_s"]
    pulse_end_s = pulse_start_s + PERTURBATION_DURATION_S

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
        label="braking pulse",
    )
    axes[0].set(title="Braking-pulse response across the 100-car platoon", ylabel="Vehicle index")
    axes[0].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0], label="Speed difference from unperturbed run (m/s)")
    axes[1].plot(times, rms_deviation, label="Fleet RMS pulse response")
    axes[1].plot(times, max_deviation, label="Maximum pulse response", alpha=0.8)
    axes[1].axvspan(
        pulse_start_s,
        pulse_end_s,
        color="black",
        alpha=0.15,
    )
    axes[1].set(xlabel="Time (s)", ylabel="Speed perturbation (m/s)")
    axes[1].legend()
    fig.savefig(output_dir / "stability_perturbation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for _, label in fd_controllers:
        selected = [row for row in fd_rows if row["controller"] == label]
        selected.sort(key=lambda row: row["density_veh_per_km"])
        ax.errorbar(
            [row["density_veh_per_km"] for row in selected],
            [row["mean_flow_veh_per_hour"] for row in selected],
            yerr=[row["ci95_flow_veh_per_hour"] for row in selected],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set(
        title="Fundamental diagram (mean ± 95% CI)",
        xlabel="Traffic density (vehicles/km)",
        ylabel="Flow (vehicles/hour)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "fundamental_diagram.png", dpi=160)
    plt.close(fig)

    # Match mean speed by interpolating across the density sweep. Density differs
    # across controllers in this diagnostic, so it is not a same-density result.
    speed_curves = {}
    for controller in ACC_VARIANTS:
        selected = [row for row in fd_rows if row["controller"] == controller.name]
        selected.sort(key=lambda row: row["mean_mean_speed_mps"])
        speed_curves[controller.name] = (
            np.asarray([row["mean_mean_speed_mps"] for row in selected]),
            np.asarray([row["mean_fuel_ml_per_vehicle_km"] for row in selected]),
            np.asarray([row["ci95_fuel_ml_per_vehicle_km"] for row in selected]),
            np.asarray([row["density_veh_per_km"] for row in selected]),
        )
    common_min = max(values[0][0] for values in speed_curves.values())
    common_max = min(values[0][-1] for values in speed_curves.values())
    matched_speeds = np.linspace(common_min, common_max, 9)
    matched_speed_rows = []
    for controller in ACC_VARIANTS:
        speeds, rates, ci95_rates, densities = speed_curves[controller.name]
        interpolated_rates = np.interp(matched_speeds, speeds, rates)
        interpolated_ci95 = np.interp(matched_speeds, speeds, ci95_rates)
        interpolated_density = np.interp(matched_speeds, speeds, densities)
        matched_speed_rows.extend(
            {
                "controller": controller.name,
                "mean_speed_mps": float(speed),
                "mean_speed_kph": float(speed * 3.6),
                "interpolated_fuel_ml_per_vehicle_km": float(rate),
                "interpolated_ci95_fuel_ml_per_vehicle_km": float(ci),
                "interpolated_density_veh_per_km": float(density),
            }
            for speed, rate, ci, density in zip(
                matched_speeds, interpolated_rates, interpolated_ci95, interpolated_density
            )
        )
    _write_csv(
        output_dir / "matched_speed_fuel.csv",
        matched_speed_rows,
        [
            "controller",
            "mean_speed_mps",
            "mean_speed_kph",
            "interpolated_fuel_ml_per_vehicle_km",
            "interpolated_ci95_fuel_ml_per_vehicle_km",
            "interpolated_density_veh_per_km",
        ],
    )
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for controller in ACC_VARIANTS:
        selected = [row for row in matched_speed_rows if row["controller"] == controller.name]
        ax.errorbar(
            [row["mean_speed_kph"] for row in selected],
            [row["interpolated_fuel_ml_per_vehicle_km"] for row in selected],
            yerr=[row["interpolated_ci95_fuel_ml_per_vehicle_km"] for row in selected],
            marker="o",
            capsize=3,
            label=controller.name,
        )
    ax.set(
        title="Fuel use at matched mean speed (density varies)",
        xlabel="Mean speed (km/h)",
        ylabel="Fuel use (mL / vehicle-km)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "matched_speed_fuel.png", dpi=160)
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
            selected = [
                row
                for row in fuel_rows
                if row["controller"] == controller.name and row["distribution"] == distribution
            ]
            selected.sort(key=lambda row: row["penetration"])
            axes[0, column].errorbar(
                [100.0 * row["penetration"] for row in selected],
                [row["mean_fuel_ml_per_vehicle_km"] for row in selected],
                yerr=[row["ci95_fuel_ml_per_vehicle_km"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
            axes[1, column].errorbar(
                [100.0 * row["penetration"] for row in selected],
                [row["mean_fuel_ml"] for row in selected],
                yerr=[row["ci95_fuel_ml"] for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
            axes[2, column].errorbar(
                [100.0 * row["penetration"] for row in selected],
                [row["mean_mean_speed_mps"] * 3.6 for row in selected],
                yerr=[row["ci95_mean_speed_mps"] * 3.6 for row in selected],
                marker="o",
                capsize=3,
                label=distribution,
            )
        axes[0, column].set_title(controller.name)
        for row_index in range(3):
            axes[row_index, column].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Fuel use\n(mL / vehicle-km)")
    axes[1, 0].set_ylabel(
        f"Fleet fuel\n(mL per {config.measurement_distance_km:g} km/vehicle)"
    )
    axes[2, 0].set_ylabel("Mean speed\n(km/h)")
    for ax in axes[-1, :]:
        ax.set_xlabel("ACC penetration (%)")
    axes[0, -1].legend(title="Distribution")
    fig.suptitle(
        "Fuel and speed at 30 vehicles/km "
        f"({config.measurement_distance_km:g} km/vehicle after "
        f"{config.warmup_distance_km:g} km warm-up; mean ± 95% CI)"
    )
    fig.savefig(output_dir / "fleet_fuel_usage.png", dpi=160)
    plt.close(fig)

    print("Saved plots and CSV summaries to:", output_dir.resolve())
    print(
        f"Each vehicle contributes exactly {config.measurement_distance_km:g} "
        "measured km after the distance-based warm-up."
    )
    print("Fuel is reported both as total mL over that common fleet distance and as mL/vehicle-km.")
    print("Each plotted point averages", replicates, "seeded runs; error bars are approximate 95% CIs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=SimulationConfig.duration,
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
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for vehicle placement.")
    parser.add_argument("--replicates", type=int, default=5, help="Number of seeded runs per scenario.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "traffic_simulation",
        help="Directory for generated plots and CSV summaries.",
    )
    parser.add_argument(
        "--fuel-coefficients",
        type=Path,
        help="JSON file with calibrated 4x4 positive_acceleration and negative_acceleration matrices.",
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
    if args.fuel_coefficients:
        fuel_model = VTMicroFuelModel.from_json(args.fuel_coefficients)
        print("Using VT-Micro coefficients loaded from:", args.fuel_coefficients)
    else:
        fuel_model = VTMicroFuelModel.provided_default()
        print("Using the supplied VT-Micro matrix for both acceleration signs.")
    run_experiments(config, fuel_model, args.output_dir, args.replicates)


if __name__ == "__main__":
    main()
