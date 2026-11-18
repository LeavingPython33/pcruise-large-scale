#!/usr/bin/env python3
"""Compare controller fuel use at matched speeds under seeded disturbances.

Run with:
    python traffic_fuel_efficiency_experiment.py --replicates 20

The controller equations and VT-Micro implementation are imported from
traffic_ring_simulation.py. Each controller starts at its own ring equilibrium;
the script does not tune or alter pcruise.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from traffic_ring_simulation import (
    ACCParameters,
    AGGRESSIVE_ACC,
    CAR_LENGTH_M,
    IDMParameters,
    N_VEHICLES,
    PCRUISE,
    VTMicroFuelModel,
    idm_acceleration,
)


CONTROLLERS: tuple[tuple[str, Optional[ACCParameters]], ...] = (
    ("human IDM", None),
    ("pcruise", PCRUISE),
    ("aggressive ACC", AGGRESSIVE_ACC),
)
DEFAULT_TARGET_SPEEDS_MPS = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
DEFAULT_WARMUP_KM = 0.1
DEFAULT_MEASUREMENT_KM = 1.3
DISTURBANCE_DISTANCE_AFTER_WARMUP_M = 50.0
DISTURBANCE_DECEL_RANGE_MPS2 = (1.5, 2.5)
DISTURBANCE_DURATION_RANGE_S = (0.6, 1.0)


@dataclass(frozen=True)
class DisturbanceProfile:
    deceleration_mps2: float
    duration_s: float


@dataclass(frozen=True)
class RunConfig:
    dt: float = 0.1
    warmup_distance_km: float = DEFAULT_WARMUP_KM
    measurement_distance_km: float = DEFAULT_MEASUREMENT_KM
    max_duration_s: float = 900.0


def equilibrium_net_gap_m(
    speed_mps: float,
    parameters: Optional[ACCParameters],
    idm: IDMParameters,
) -> float:
    """Return the steady net gap for a controller at the requested speed."""
    if speed_mps <= 0.0:
        raise ValueError("Target speeds must be positive.")
    if parameters is not None:
        return parameters.g_min + parameters.tau * speed_mps

    free_road_term = (speed_mps / idm.desired_speed) ** idm.exponent
    if free_road_term >= 1.0:
        raise ValueError("IDM target speed must be below its desired speed.")
    dynamic_gap = idm.minimum_gap + speed_mps * idm.time_headway
    return dynamic_gap / np.sqrt(1.0 - free_road_term)


def draw_disturbance(seed: int) -> DisturbanceProfile:
    """Draw a mild braking pulse; the same seed is paired across controllers."""
    rng = np.random.default_rng(seed)
    return DisturbanceProfile(
        deceleration_mps2=float(rng.uniform(*DISTURBANCE_DECEL_RANGE_MPS2)),
        duration_s=float(rng.uniform(*DISTURBANCE_DURATION_RANGE_S)),
    )


def simulate_ring_run(
    *,
    controller_name: str,
    controller_parameters: Optional[ACCParameters],
    target_speed_mps: float,
    config: RunConfig,
    fuel_model: VTMicroFuelModel,
    idm: IDMParameters,
    disturbance: Optional[DisturbanceProfile],
) -> dict:
    """Simulate a homogeneous ring from the selected controller's equilibrium."""
    net_gap_m = equilibrium_net_gap_m(target_speed_mps, controller_parameters, idm)
    center_spacing_m = net_gap_m + CAR_LENGTH_M
    density_veh_per_km = 1000.0 / center_spacing_m
    road_length_m = N_VEHICLES * center_spacing_m
    positions = np.arange(N_VEHICLES, dtype=float) * center_spacing_m
    speeds = np.full(N_VEHICLES, target_speed_mps, dtype=float)
    vehicle_distance_m = np.zeros(N_VEHICLES, dtype=float)
    measured_distance_m = np.zeros(N_VEHICLES, dtype=float)
    measured_time_s = np.zeros(N_VEHICLES, dtype=float)
    fuel_by_vehicle_ml = np.zeros(N_VEHICLES, dtype=float)
    speed_time_integral = 0.0
    minimum_gap_m = float("inf")
    collision_steps = 0

    warmup_m = config.warmup_distance_km * 1000.0
    measurement_end_m = warmup_m + config.measurement_distance_km * 1000.0
    disturbance_trigger_m = warmup_m + DISTURBANCE_DISTANCE_AFTER_WARMUP_M
    disturbance_start_s: Optional[float] = None
    idm_parameters = controller_parameters is None
    max_steps = int(np.ceil(config.max_duration_s / config.dt))

    for step in range(max_steps):
        time_s = step * config.dt
        leaders = np.roll(positions, -1)
        gaps = (leaders - positions) % road_length_m - CAR_LENGTH_M
        minimum_gap_m = min(minimum_gap_m, float(np.min(gaps)))
        collision_steps += int(np.count_nonzero(gaps <= 0.0))
        safe_gaps = np.maximum(gaps, 0.1)
        leader_speeds = np.roll(speeds, -1)

        if idm_parameters:
            acceleration = idm_acceleration(safe_gaps, speeds, leader_speeds, idm)
        else:
            assert controller_parameters is not None
            desired_gaps = (
                controller_parameters.g_min + controller_parameters.tau * speeds
            )
            command = (
                controller_parameters.alpha * (safe_gaps - desired_gaps)
                + controller_parameters.lam * (leader_speeds - speeds)
            )
            acceleration = np.clip(
                command,
                controller_parameters.a_min,
                controller_parameters.a_max,
            )

        if (
            disturbance is not None
            and disturbance_start_s is None
            and vehicle_distance_m[0] >= disturbance_trigger_m
        ):
            disturbance_start_s = time_s
        if (
            disturbance is not None
            and disturbance_start_s is not None
            and disturbance_start_s <= time_s < disturbance_start_s + disturbance.duration_s
        ):
            acceleration[0] = min(
                acceleration[0], -disturbance.deceleration_mps2
            )

        next_speeds = np.maximum(0.0, speeds + acceleration * config.dt)
        step_distance_m = 0.5 * (speeds + next_speeds) * config.dt
        next_positions = (positions + step_distance_m) % road_length_m
        next_vehicle_distance_m = vehicle_distance_m + step_distance_m
        measured_step_distance_m = np.maximum(
            0.0,
            np.minimum(next_vehicle_distance_m, measurement_end_m)
            - np.maximum(vehicle_distance_m, warmup_m),
        )
        measurement_fraction = np.divide(
            measured_step_distance_m,
            step_distance_m,
            out=np.zeros_like(measured_step_distance_m),
            where=step_distance_m > 0.0,
        )
        measured_step_time_s = config.dt * measurement_fraction
        fuel_rate_ml_s = fuel_model.rate_ml_per_s(next_speeds, acceleration)

        fuel_by_vehicle_ml += fuel_rate_ml_s * measured_step_time_s
        measured_distance_m += measured_step_distance_m
        measured_time_s += measured_step_time_s
        speed_time_integral += float(np.sum(next_speeds * measured_step_time_s))

        positions, speeds = next_positions, next_speeds
        vehicle_distance_m = next_vehicle_distance_m
        if np.all(vehicle_distance_m >= measurement_end_m):
            break
    else:
        completed = int(np.count_nonzero(vehicle_distance_m >= measurement_end_m))
        raise RuntimeError(
            f"{controller_name} at {target_speed_mps:g} m/s completed only "
            f"{completed}/{N_VEHICLES} distance windows."
        )

    total_distance_km = float(np.sum(measured_distance_m) / 1000.0)
    total_time_s = float(np.sum(measured_time_s))
    mean_speed_mps = speed_time_integral / total_time_s
    total_fuel_ml = float(np.sum(fuel_by_vehicle_ml))
    return {
        "controller": controller_name,
        "target_speed_mps": target_speed_mps,
        "density_veh_per_km": density_veh_per_km,
        "mean_speed_mps": mean_speed_mps,
        "flow_veh_per_h": density_veh_per_km * mean_speed_mps * 3.6,
        "fuel_ml_per_vehicle_km": total_fuel_ml / total_distance_km,
        "total_fuel_ml": total_fuel_ml,
        "total_vehicle_km": total_distance_km,
        "minimum_gap_m": minimum_gap_m,
        "collision_steps": collision_steps,
        "disturbance_deceleration_mps2": (
            disturbance.deceleration_mps2 if disturbance is not None else 0.0
        ),
        "disturbance_duration_s": (
            disturbance.duration_s if disturbance is not None else 0.0
        ),
    }


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half_width, mean + half_width


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_controller_curves(
    rows: list[dict],
    value_key: str,
    response_key: str,
    replicate_count: int,
) -> tuple[float, float, np.ndarray, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """Build paired replicate curves over the common response range."""
    valid_rows = [row for row in rows if row["collision_steps"] == 0]
    target_count = len({row["target_speed_mps"] for row in rows})
    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for name, _ in CONTROLLERS:
        curves[name] = {}
        for replicate in range(replicate_count):
            selected = [
                row for row in valid_rows
                if row["controller"] == name and row["replicate"] == replicate
            ]
            selected.sort(key=lambda row: row[response_key])
            if len(selected) == target_count:
                x = np.asarray([row[response_key] for row in selected], dtype=float)
                y = np.asarray([row[value_key] for row in selected], dtype=float)
                unique_x, unique_indices = np.unique(x, return_index=True)
                curves[name][replicate] = (unique_x, y[unique_indices])

    complete_replicates = [
        replicate
        for replicate in range(replicate_count)
        if all(replicate in curves[name] for name, _ in CONTROLLERS)
    ]
    if not complete_replicates:
        raise RuntimeError("No complete, collision-free paired replicates are available.")

    lower = max(
        curves[name][replicate][0][0]
        for name, _ in CONTROLLERS
        for replicate in complete_replicates
    )
    upper = min(
        curves[name][replicate][0][-1]
        for name, _ in CONTROLLERS
        for replicate in complete_replicates
    )
    if upper <= lower:
        raise RuntimeError(
            f"No common {response_key} range across controllers "
            f"({lower:.3g} to {upper:.3g})."
        )
    common_values = np.linspace(lower, upper, 8)
    return lower, upper, common_values, {
        name: {replicate: curves[name][replicate] for replicate in complete_replicates}
        for name, _ in CONTROLLERS
    }


def analyze_matched_response(
    rows: list[dict],
    *,
    response_key: str,
    response_label: str,
    output_dir: Path,
    replicate_count: int,
) -> list[dict]:
    lower, upper, common_values, curves = make_controller_curves(
        rows, "fuel_ml_per_vehicle_km", response_key, replicate_count
    )
    replicates = sorted(set.intersection(*[
        set(curves[name]) for name, _ in CONTROLLERS
    ]))
    evaluated = {
        name: {
            replicate: np.interp(common_values, *curves[name][replicate])
            for replicate in replicates
        }
        for name, _ in CONTROLLERS
    }
    comparison_rows: list[dict] = []
    for comparator in ("human IDM", "aggressive ACC"):
        for index, value in enumerate(common_values):
            differences = np.asarray(
                [
                    evaluated["pcruise"][replicate][index]
                    - evaluated[comparator][replicate][index]
                    for replicate in replicates
                ],
                dtype=float,
            )
            delta, ci_low, ci_high = mean_ci(differences)
            comparison_rows.append(
                {
                    response_key: float(value),
                    "comparison": f"pcruise minus {comparator}",
                    "mean_fuel_difference_ml_per_vehicle_km": delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "replicates": len(differences),
                    "share_replicates_pcruise_lower": float(np.mean(differences < 0.0)),
                }
            )
    write_csv(output_dir / f"matched_{response_key}_comparison.csv", comparison_rows)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = {"human IDM": "#4c78a8", "pcruise": "#e45756", "aggressive ACC": "#54a24b"}
    for name, _ in CONTROLLERS:
        samples = np.asarray([evaluated[name][replicate] for replicate in replicates])
        means = np.mean(samples, axis=0)
        half_width = (
            1.96 * np.std(samples, axis=0, ddof=1) / np.sqrt(len(replicates))
            if len(replicates) > 1
            else np.zeros_like(means)
        )
        axis.plot(common_values, means, marker="o", label=name, color=colors[name])
        axis.fill_between(
            common_values,
            means - half_width,
            means + half_width,
            color=colors[name],
            alpha=0.16,
        )
    axis.set_xlabel(response_label)
    axis.set_ylabel("Fuel (mL / vehicle-km)")
    axis.set_title(f"Fuel at matched {response_label.lower()}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"fuel_vs_matched_{response_key}.png", dpi=180)
    plt.close(figure)

    delta_figure, delta_axis = plt.subplots(figsize=(9, 5.5))
    for comparator, color in (("human IDM", colors["human IDM"]), ("aggressive ACC", colors["aggressive ACC"])):
        selected = [
            row for row in comparison_rows
            if row["comparison"] == f"pcruise minus {comparator}"
        ]
        delta_axis.plot(
            common_values,
            [row["mean_fuel_difference_ml_per_vehicle_km"] for row in selected],
            marker="o",
            label=f"pcruise minus {comparator}",
            color=color,
        )
        delta_axis.fill_between(
            common_values,
            [row["ci95_low"] for row in selected],
            [row["ci95_high"] for row in selected],
            color=color,
            alpha=0.16,
        )
    delta_axis.axhline(0.0, color="black", linewidth=1)
    delta_axis.set_xlabel(response_label)
    delta_axis.set_ylabel("Fuel difference (mL / vehicle-km)")
    delta_axis.set_title("Negative values favor pcruise")
    delta_axis.grid(True, alpha=0.3)
    delta_axis.legend()
    delta_figure.tight_layout()
    delta_figure.savefig(output_dir / f"fuel_difference_vs_matched_{response_key}.png", dpi=180)
    plt.close(delta_figure)
    print(
        f"Matched {response_label.lower()} range: {lower:.2f} to {upper:.2f}; "
        f"{len(replicates)} paired, collision-free replicates."
    )
    return comparison_rows


def write_operating_point_plot(rows: list[dict], output_dir: Path) -> list[dict]:
    summary_rows: list[dict] = []
    for name, _ in CONTROLLERS:
        for target_speed in sorted({row["target_speed_mps"] for row in rows}):
            samples = [
                row for row in rows
                if row["controller"] == name
                and row["target_speed_mps"] == target_speed
                and row["collision_steps"] == 0
            ]
            if not samples:
                continue
            for key, prefix in (
                ("mean_speed_mps", "mean_speed"),
                ("flow_veh_per_h", "flow"),
                ("fuel_ml_per_vehicle_km", "fuel_ml_per_vehicle_km"),
            ):
                values = np.asarray([row[key] for row in samples], dtype=float)
                mean, low, high = mean_ci(values)
                if key == "mean_speed_mps":
                    record = {
                        "controller": name,
                        "target_speed_mps": target_speed,
                        "valid_replicates": len(samples),
                    }
                record[f"{prefix}_mean"] = mean
                record[f"{prefix}_ci95_low"] = low
                record[f"{prefix}_ci95_high"] = high
            summary_rows.append(record)
    write_csv(output_dir / "operating_points.csv", summary_rows)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = {"human IDM": "#4c78a8", "pcruise": "#e45756", "aggressive ACC": "#54a24b"}
    for name, _ in CONTROLLERS:
        selected = [row for row in summary_rows if row["controller"] == name]
        axis.errorbar(
            [row["flow_mean"] for row in selected],
            [row["fuel_ml_per_vehicle_km_mean"] for row in selected],
            xerr=[
                [row["flow_mean"] - row["flow_ci95_low"] for row in selected],
                [row["flow_ci95_high"] - row["flow_mean"] for row in selected],
            ],
            yerr=[
                [
                    row["fuel_ml_per_vehicle_km_mean"] - row["fuel_ml_per_vehicle_km_ci95_low"]
                    for row in selected
                ],
                [
                    row["fuel_ml_per_vehicle_km_ci95_high"] - row["fuel_ml_per_vehicle_km_mean"]
                    for row in selected
                ],
            ],
            marker="o",
            capsize=3,
            label=name,
            color=colors[name],
        )
    axis.set_xlabel("Achieved flow (vehicles/hour)")
    axis.set_ylabel("Fuel (mL / vehicle-km)")
    axis.set_title("Fuel-flow operating points (not matched-flow comparisons)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fuel_vs_flow_operating_points.png", dpi=180)
    plt.close(figure)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--warmup-km", type=float, default=DEFAULT_WARMUP_KM)
    parser.add_argument("--distance-km", type=float, default=DEFAULT_MEASUREMENT_KM)
    parser.add_argument(
        "--target-speeds-mps",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_SPEEDS_MPS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "fuel_efficiency_experiment",
    )
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError("Use at least two replicates to estimate uncertainty.")
    if args.dt <= 0.0 or args.distance_km <= 0.0 or args.warmup_km < 0.0:
        raise ValueError("dt and distance must be positive; warm-up must be nonnegative.")
    if not args.target_speeds_mps or any(speed <= 0.0 for speed in args.target_speeds_mps):
        raise ValueError("Provide positive target speeds.")
    if len(set(args.target_speeds_mps)) != len(args.target_speeds_mps):
        raise ValueError("Target speeds must be unique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        dt=args.dt,
        warmup_distance_km=args.warmup_km,
        measurement_distance_km=args.distance_km,
    )
    fuel_model = VTMicroFuelModel.provided_default()
    idm = IDMParameters()
    run_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for target_speed in sorted(args.target_speeds_mps):
        baseline_at_speed = []
        for controller_name, controller_parameters in CONTROLLERS:
            result = simulate_ring_run(
                controller_name=controller_name,
                controller_parameters=controller_parameters,
                target_speed_mps=target_speed,
                config=config,
                fuel_model=fuel_model,
                idm=idm,
                disturbance=None,
            )
            baseline_at_speed.append(result["fuel_ml_per_vehicle_km"])
            baseline_rows.append(
                {
                    "scenario": "steady_state_no_disturbance",
                    "replicate": -1,
                    "seed": -1,
                    **result,
                }
            )
        if not np.allclose(baseline_at_speed, baseline_at_speed[0], rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                f"Steady-state fuel rates did not agree at {target_speed:g} m/s; "
                "check equilibrium initialization and fuel integration."
            )

    for replicate in range(args.replicates):
        seed = args.seed + replicate
        disturbance = draw_disturbance(seed)
        for target_speed in sorted(args.target_speeds_mps):
            for controller_name, controller_parameters in CONTROLLERS:
                result = simulate_ring_run(
                    controller_name=controller_name,
                    controller_parameters=controller_parameters,
                    target_speed_mps=target_speed,
                    config=config,
                    fuel_model=fuel_model,
                    idm=idm,
                    disturbance=disturbance,
                )
                run_rows.append(
                    {
                        "scenario": "seeded_braking_pulse",
                        "replicate": replicate,
                        "seed": seed,
                        **result,
                    }
                )
    write_csv(args.output_dir / "fuel_efficiency_runs.csv", run_rows)
    write_csv(args.output_dir / "steady_state_baselines.csv", baseline_rows)
    operating_points = write_operating_point_plot(run_rows, args.output_dir)
    matched_speed_rows = analyze_matched_response(
        run_rows,
        response_key="mean_speed_mps",
        response_label="achieved mean speed (m/s)",
        output_dir=args.output_dir,
        replicate_count=args.replicates,
    )

    flow_values = np.asarray([row["flow_mean"] for row in operating_points], dtype=float)
    if flow_values.size:
        controller_flow_ranges = []
        for name, _ in CONTROLLERS:
            values = np.asarray(
                [row["flow_mean"] for row in operating_points if row["controller"] == name],
                dtype=float,
            )
            if values.size:
                controller_flow_ranges.append((float(np.min(values)), float(np.max(values))))
        common_flow_low = max(bounds[0] for bounds in controller_flow_ranges)
        common_flow_high = min(bounds[1] for bounds in controller_flow_ranges)
        if common_flow_high > common_flow_low:
            flow_rows = analyze_matched_response(
                run_rows,
                response_key="flow_veh_per_h",
                response_label="achieved flow (vehicles/hour)",
                output_dir=args.output_dir,
                replicate_count=args.replicates,
            )
        else:
            flow_rows = []
            print(
                "No common achieved-flow range across all three controllers in this "
                "speed sweep; no matched-flow comparison was extrapolated."
            )
        write_csv(
            args.output_dir / "matched_flow_status.csv",
            [
                {
                    "status": (
                        "available"
                        if flow_rows
                        else "no common achieved-flow range; do not extrapolate"
                    ),
                    "common_flow_low_veh_per_h": common_flow_low,
                    "common_flow_high_veh_per_h": common_flow_high,
                }
            ],
        )

    collisions = sum(row["collision_steps"] > 0 for row in run_rows)
    print(f"Saved results to: {args.output_dir.resolve()}")
    print(
        f"Ran {len(run_rows)} paired disturbance runs; "
        f"{collisions} runs contained at least one overlap step."
    )
    print(
        "Matched-speed fuel differences are paired by disturbance seed. "
        "Negative pcruise-minus-comparator values favor pcruise."
    )
    print(
        "Fuel uses the supplied VT-Micro coefficients for both acceleration "
        "signs; validate separate sign-specific matrices before treating values "
        "as vehicle predictions."
    )
    if matched_speed_rows:
        supported = sum(
            row["ci95_high"] < 0.0 for row in matched_speed_rows
        )
        print(
            f"Matched-speed points with a 95% CI entirely favoring pcruise: "
            f"{supported}/{len(matched_speed_rows)}."
        )


if __name__ == "__main__":
    main()
