#!/usr/bin/env python3
"""Generate a time-space diagram for a mixed-traffic straight-road platoon.

The 30-car finite platoon uses delayed, noisy IDM for human drivers and the
specified pcruise proportional-derivative law for ACC vehicles. Two human
drivers brake toward 20 mph at 15 s and 45 s.

Requirements:
    python -m pip install numpy matplotlib

Run:
    python traffic_time_space_simulation.py
    python traffic_time_space_simulation.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MPH_TO_MPS = 0.44704
PCRUISE_ALPHA = 0.8
PCRUISE_TAU_S = 2.5
PCRUISE_G_MIN_M = 2.0
PCRUISE_LAMBDA = 0.05
PCRUISE_A_MIN_MPS2 = -5.0
PCRUISE_A_MAX_MPS2 = 4.0


@dataclass(frozen=True)
class Disturbance:
    """A forced braking event for one human-driven car (one-based ID)."""

    time_s: float
    vehicle_id: int


@dataclass(frozen=True)
class SimulationConfig:
    n_vehicles: int = 30
    duration_s: float = 90.0
    dt_s: float = 0.1
    vehicle_length_m: float = 4.5
    initial_spacing_m: float = 74.0
    initial_speed_mps: float = 60.0 * MPH_TO_MPS
    desired_speed_mps: float = 64.0 * MPH_TO_MPS
    human_reaction_s: float = 0.9
    human_jitter_std_mps: float = 0.18
    human_jitter_correlation_s: float = 1.5
    event_target_speed_mps: float = 20.0 * MPH_TO_MPS
    event_brake_deceleration_mps2: float = 5.0
    event_duration_s: float = 8.0
    seed: int = 2026

    @property
    def disturbances(self) -> tuple[Disturbance, ...]:
        return (Disturbance(15.0, 23), Disturbance(45.0, 8))


def build_fleet(n_vehicles: int) -> np.ndarray:
    """Return a regularly interleaved fleet mask; True means ACC-equipped."""
    vehicle_indices = np.arange(n_vehicles)
    # Two ACC vehicles in every five-car group gives 40% ACC penetration.
    acc_mask = np.isin(vehicle_indices % 5, (1, 3))
    for event in (Disturbance(15.0, 23), Disturbance(45.0, 8)):
        if event.vehicle_id > n_vehicles or acc_mask[event.vehicle_id - 1]:
            raise ValueError("Disturbance vehicle IDs must identify human drivers.")
    return acc_mask


def idm_acceleration(
    speed_mps: float,
    leader_speed_mps: float,
    net_gap_m: float,
    desired_speed_mps: float,
    *,
    minimum_gap_m: float,
    time_headway_s: float,
    max_acceleration_mps2: float,
    comfortable_deceleration_mps2: float,
) -> float:
    """Compute one standard IDM acceleration command."""
    gap = max(net_gap_m, 0.1)
    closing_speed = speed_mps - leader_speed_mps
    dynamic_gap = minimum_gap_m + max(
        0.0,
        speed_mps * time_headway_s
        + speed_mps * closing_speed
        / (2.0 * np.sqrt(max_acceleration_mps2 * comfortable_deceleration_mps2)),
    )
    free_road_term = (speed_mps / max(desired_speed_mps, 0.1)) ** 4
    interaction_term = (dynamic_gap / gap) ** 2
    command = max_acceleration_mps2 * (1.0 - free_road_term - interaction_term)
    return float(np.clip(command, -5.0, 2.5))


def pcruise_acceleration(
    actual_gap_m: float,
    ego_speed_mps: float,
    leader_speed_mps: float,
) -> float:
    """Apply the specified pcruise PD law and its acceleration saturation."""
    desired_gap_m = PCRUISE_G_MIN_M + PCRUISE_TAU_S * ego_speed_mps
    spacing_error_m = actual_gap_m - desired_gap_m
    relative_speed_mps = leader_speed_mps - ego_speed_mps
    command_mps2 = (
        PCRUISE_ALPHA * spacing_error_m
        + PCRUISE_LAMBDA * relative_speed_mps
    )
    return float(
        np.clip(command_mps2, PCRUISE_A_MIN_MPS2, PCRUISE_A_MAX_MPS2)
    )


def simulate(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return position history (m) and the ACC mask for the platoon."""
    if config.n_vehicles < 25:
        raise ValueError("Use at least 25 vehicles so both event-car IDs are valid.")
    if config.dt_s <= 0.0 or config.duration_s <= 45.0 + config.event_duration_s:
        raise ValueError("Use a positive time step and duration greater than 49.5 s.")
    if config.initial_spacing_m <= config.vehicle_length_m:
        raise ValueError("Initial spacing must exceed vehicle length.")

    rng = np.random.default_rng(config.seed)
    acc_mask = build_fleet(config.n_vehicles)
    human_mask = ~acc_mask
    step_count = int(round(config.duration_s / config.dt_s))
    time_s = np.arange(step_count + 1, dtype=float) * config.dt_s

    positions = np.zeros((step_count + 1, config.n_vehicles), dtype=float)
    speeds = np.full((step_count + 1, config.n_vehicles), config.initial_speed_mps)
    positions[0] = np.arange(config.n_vehicles) * config.initial_spacing_m
    virtual_lead_positions = (
        positions[0, -1]
        + config.initial_spacing_m
        + config.initial_speed_mps * time_s
    )
    human_delays = rng.uniform(
        max(0.0, config.human_reaction_s - 0.15),
        config.human_reaction_s + 0.15,
        size=config.n_vehicles,
    )
    delay_steps = np.rint(human_delays / config.dt_s).astype(int)
    speed_preference_noise = np.zeros(config.n_vehicles, dtype=float)
    event_start_indices = {
        event.vehicle_id - 1: int(round(event.time_s / config.dt_s))
        for event in config.disturbances
    }

    for step in range(step_count):
        accelerations = np.zeros(config.n_vehicles, dtype=float)

        # Correlated low-amplitude speed-preference variation models small,
        # natural human-driver inconsistencies without white-noise speed jumps.
        noise_step = (
            -speed_preference_noise[human_mask]
            * config.dt_s
            / config.human_jitter_correlation_s
            + config.human_jitter_std_mps
            * np.sqrt(2.0 * config.dt_s / config.human_jitter_correlation_s)
            * rng.standard_normal(np.count_nonzero(human_mask))
        )
        speed_preference_noise[human_mask] += noise_step

        for vehicle in range(config.n_vehicles):
            if acc_mask[vehicle]:
                observed_step = step
            else:
                observed_step = max(0, step - delay_steps[vehicle])
            if vehicle + 1 < config.n_vehicles:
                leader_position = positions[observed_step, vehicle + 1]
                leader_speed = speeds[observed_step, vehicle + 1]
            else:
                leader_position = virtual_lead_positions[observed_step]
                leader_speed = config.initial_speed_mps

            net_gap = (
                leader_position
                - positions[observed_step, vehicle]
                - config.vehicle_length_m
            )

            if acc_mask[vehicle]:
                accelerations[vehicle] = pcruise_acceleration(
                    net_gap,
                    speeds[observed_step, vehicle],
                    leader_speed,
                )
            else:
                desired_speed = max(
                    5.0,
                    config.desired_speed_mps + speed_preference_noise[vehicle],
                )
                accelerations[vehicle] = idm_acceleration(
                    speeds[observed_step, vehicle],
                    leader_speed,
                    net_gap,
                    desired_speed,
                    minimum_gap_m=2.0,
                    time_headway_s=1.2,
                    max_acceleration_mps2=1.2,
                    comfortable_deceleration_mps2=2.0,
                )

        # Scheduled human-only braking overrides ordinary car-following control.
        forced_braking = np.zeros(config.n_vehicles, dtype=bool)
        for vehicle, start_step in event_start_indices.items():
            end_step = start_step + int(round(config.event_duration_s / config.dt_s))
            if start_step <= step < end_step:
                forced_braking[vehicle] = True
                speed = speeds[step, vehicle]
                if speed > config.event_target_speed_mps:
                    accelerations[vehicle] = max(
                        -config.event_brake_deceleration_mps2,
                        (config.event_target_speed_mps - speed) / config.dt_s,
                    )
                else:
                    accelerations[vehicle] = 0.0

        next_speeds = np.maximum(0.0, speeds[step] + accelerations * config.dt_s)
        for vehicle in np.flatnonzero(forced_braking):
            next_speeds[vehicle] = max(
                config.event_target_speed_mps,
                next_speeds[vehicle],
            )
            if speeds[step, vehicle] <= config.event_target_speed_mps:
                next_speeds[vehicle] = config.event_target_speed_mps

        positions[step + 1] = positions[step] + (
            0.5 * (speeds[step] + next_speeds) * config.dt_s
        )
        speeds[step + 1] = next_speeds

    net_gaps = positions[:, 1:] - positions[:, :-1] - config.vehicle_length_m
    if np.any(net_gaps < 0.0):
        raise RuntimeError(
            "The simulated trajectories overlap; increase initial spacing or revise "
            "reaction/braking assumptions."
        )
    return positions, acc_mask


def plot_time_space(
    config: SimulationConfig,
    positions_m: np.ndarray,
    acc_mask: np.ndarray,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    """Plot and save high-resolution vehicle trajectories and brake events."""
    times = np.arange(positions_m.shape[0], dtype=float) * config.dt_s
    positions_km = positions_m / 1000.0
    human_color = "#c55a11"
    acc_color = "#087e8b"

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    for vehicle in range(config.n_vehicles):
        is_acc = bool(acc_mask[vehicle])
        ax.plot(
            times,
            positions_km[:, vehicle],
            color=acc_color if is_acc else human_color,
            linestyle="-" if is_acc else "--",
            linewidth=1.65 if is_acc else 1.35,
            alpha=0.9 if is_acc else 0.76,
            zorder=3 if is_acc else 2,
        )

    annotation_offsets = ((24, -45), (24, 35))
    for event_number, event in enumerate(config.disturbances):
        step = int(round(event.time_s / config.dt_s))
        event_position_km = positions_km[step, event.vehicle_id - 1]
        pre_event_speed_mph = (
            (positions_m[step, event.vehicle_id - 1] - positions_m[step - 1, event.vehicle_id - 1])
            / config.dt_s
            / MPH_TO_MPS
        )
        ax.axvline(event.time_s, color="#555555", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.scatter(
            [event.time_s],
            [event_position_km],
            marker="X",
            s=95,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"Human vehicle {event.vehicle_id}: "
            f"{pre_event_speed_mph:.0f} to 20 mph braking\n"
            f"t = {event.time_s:g} s, x = {event_position_km:.2f} km",
            xy=(event.time_s, event_position_km),
            xytext=annotation_offsets[event_number],
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#999999", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9},
            zorder=6,
        )

    ax.set(
        title=(
            "Mixed-traffic time-space diagram: human braking disturbances "
            "in a 30-vehicle platoon"
        ),
        xlabel="Time (s)",
        ylabel="Position along straight road (km; rear to front)",
        xlim=(0.0, config.duration_s),
    )
    ax.grid(True, which="major", color="#aab2b8", alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], color=human_color, linestyle="--", linewidth=1.8,
                   label="Human driver (delayed, stochastic IDM)"),
            Line2D(
                [0], [0], color=acc_color, linestyle="-", linewidth=1.8,
                label="pcruise ACC (PD controller)",
            ),
            Line2D([0], [0], color="#d62728", marker="X", linestyle="None",
                   markersize=8, label="Scheduled human braking event"),
        ],
        loc="best",
        framealpha=0.95,
    )
    fig.text(
        0.01,
        0.005,
        "Synthetic model: human delays and speed variation are illustrative; "
        "pcruise uses its specified PD law and saturation.",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "time_space_diagram.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution in dots per inch.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for human-driver noise.")
    parser.add_argument("--show", action="store_true", help="Display the plot after saving it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 100:
        raise ValueError("Use at least 100 DPI for a readable output image.")
    config = SimulationConfig(seed=args.seed)
    positions_m, acc_mask = simulate(config)
    plot_time_space(config, positions_m, acc_mask, args.output, args.dpi, args.show)
    print(f"Saved {config.n_vehicles} vehicle trajectories to {args.output.resolve()}")
    print(
        f"Fleet: {np.count_nonzero(acc_mask)} ACC, "
        f"{np.count_nonzero(~acc_mask)} human drivers."
    )
    for event in config.disturbances:
        event_step = int(round(event.time_s / config.dt_s))
        pre_event_speed_mph = (
            (positions_m[event_step, event.vehicle_id - 1]
             - positions_m[event_step - 1, event.vehicle_id - 1])
            / config.dt_s
            / MPH_TO_MPS
        )
        print(
            f"Brake event: human vehicle {event.vehicle_id}, "
            f"{pre_event_speed_mph:.1f} to 20 mph at {event.time_s:g} s, "
            f"position {positions_m[event_step, event.vehicle_id - 1] / 1000.0:.3f} km."
        )


if __name__ == "__main__":
    main()
