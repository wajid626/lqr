from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are


# -----------------------------
# Vehicle and simulation settings
# -----------------------------

WHEELBASE = 2.8  # m
SPEED = 5.0  # m/s, constant longitudinal speed
DT = 0.05  # s
MAX_STEER_DEG = 35.0
SIM_TIME = 18.0

# 4-state dynamic bicycle model parameters
M = 1500.0  # kg
IZ = 2250.0  # kg m^2
LF = 1.2  # m
LR = 1.6  # m
CF = 80000.0  # N/rad
CR = 80000.0  # N/rad
VX = SPEED


@dataclass(frozen=True)
class LQRCase:
    name: str
    q_diag: tuple[float, float, float, float]
    r: float


@dataclass
class VehicleState:
    x: float
    y: float
    theta: float
    vy: float = 0.0
    r: float = 0.0


def continuous_bicycle_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return the continuous-time 4-state lateral bicycle model.

    State x = [e, psi_e, v_y, r]^T and input u = delta.
    """

    a33 = -(CF + CR) / (M * VX)
    a34 = (-CF * LF + CR * LR) / (M * VX) - VX
    a43 = (-CF * LF + CR * LR) / (IZ * VX)
    a44 = -(CF * LF**2 + CR * LR**2) / (IZ * VX)

    a_c = np.array(
        [
            [0.0, VX, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, a33, a34],
            [0.0, 0.0, a43, a44],
        ],
        dtype=float,
    )
    b_c = np.array([[0.0], [0.0], [CF / M], [CF * LF / IZ]], dtype=float)
    return a_c, b_c


def discretize_euler(a_c: np.ndarray, b_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    return np.eye(a_c.shape[0]) + a_c * dt, b_c * dt


def lqr_gain(q_diag: tuple[float, float, float, float], r_value: float) -> np.ndarray:
    a_c, b_c = continuous_bicycle_matrices()
    a_d, b_d = discretize_euler(a_c, b_c, DT)
    q = np.diag(q_diag)
    r = np.array([[r_value]], dtype=float)
    p = solve_discrete_are(a_d, b_d, q, r)
    return np.linalg.solve(b_d.T @ p @ b_d + r, b_d.T @ p @ a_d)


def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def path_polynomial_sine(x_values: np.ndarray) -> np.ndarray:
    return 0.02 * x_values + 0.004 * x_values**2 - 0.00004 * x_values**3 + 6.0 * np.sin(0.12 * x_values)


def path_sine(x_values: np.ndarray) -> np.ndarray:
    return 4.0 * np.sin(0.18 * x_values)


def path_lane_change(x_values: np.ndarray) -> np.ndarray:
    return 3.2 * (np.tanh((x_values - 25.0) / 6.0) - np.tanh((x_values - 55.0) / 6.0))


def build_path(path_function, x_end: float = 90.0, spacing: float = 0.5) -> dict[str, np.ndarray]:
    x = np.arange(0.0, x_end + spacing, spacing)
    y = path_function(x)

    dy_dx = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    heading = np.arctan2(dy_dx, np.ones_like(dy_dx))
    curvature = d2y_dx2 / np.maximum((1.0 + dy_dx**2) ** 1.5, 1e-9)

    return {"x": x, "y": y, "heading": heading, "curvature": curvature}


def nearest_path_index(state: VehicleState, path: dict[str, np.ndarray], start_index: int) -> int:
    search_end = min(start_index + 45, len(path["x"]))
    dx = path["x"][start_index:search_end] - state.x
    dy = path["y"][start_index:search_end] - state.y
    if len(dx) == 0:
        return len(path["x"]) - 1
    return start_index + int(np.argmin(dx * dx + dy * dy))


def tracking_errors(state: VehicleState, path: dict[str, np.ndarray], index: int) -> tuple[float, float]:
    dx = state.x - float(path["x"][index])
    dy = state.y - float(path["y"][index])
    heading = float(path["heading"][index])
    e = -math.sin(heading) * dx + math.cos(heading) * dy
    psi_e = wrap_to_pi(state.theta - heading)
    return e, psi_e


def control_law(
    state: VehicleState,
    path: dict[str, np.ndarray],
    index: int,
    gain: np.ndarray,
    use_feedforward: bool = True,
) -> tuple[float, float, float]:
    e, psi_e = tracking_errors(state, path, index)
    x_lqr = np.array([[e], [psi_e], [state.vy], [state.r]], dtype=float)

    delta_feedback = float((-gain @ x_lqr).item())
    delta_feedforward = math.atan(WHEELBASE * float(path["curvature"][index])) if use_feedforward else 0.0
    raw_delta = delta_feedforward + delta_feedback
    delta = clamp(raw_delta, -math.radians(MAX_STEER_DEG), math.radians(MAX_STEER_DEG))
    return delta, e, psi_e


def update_vehicle_state(state: VehicleState, steering_rad: float) -> VehicleState:
    a_c, b_c = continuous_bicycle_matrices()
    vy_dot = a_c[2, 2] * state.vy + a_c[2, 3] * state.r + float(b_c[2, 0]) * steering_rad
    r_dot = a_c[3, 2] * state.vy + a_c[3, 3] * state.r + float(b_c[3, 0]) * steering_rad

    x_dot = VX * math.cos(state.theta) - state.vy * math.sin(state.theta)
    y_dot = VX * math.sin(state.theta) + state.vy * math.cos(state.theta)
    theta_dot = state.r

    return VehicleState(
        x=state.x + x_dot * DT,
        y=state.y + y_dot * DT,
        theta=wrap_to_pi(state.theta + theta_dot * DT),
        vy=state.vy + vy_dot * DT,
        r=state.r + r_dot * DT,
    )


def simulate_path(path: dict[str, np.ndarray], case: LQRCase, lateral_offset: float = -0.25) -> dict[str, np.ndarray | float | str]:
    gain = lqr_gain(case.q_diag, case.r)
    state = VehicleState(
        x=float(path["x"][0]),
        y=float(path["y"][0] + lateral_offset),
        theta=float(path["heading"][0]),
    )

    n_steps = int(SIM_TIME / DT)
    index = 0
    history: dict[str, list[float]] = {
        "t": [],
        "x": [],
        "y": [],
        "theta": [],
        "e": [],
        "psi_e": [],
        "steer": [],
        "vy": [],
        "r": [],
    }

    for step in range(n_steps):
        index = nearest_path_index(state, path, index)
        steering, e, psi_e = control_law(state, path, index, gain)

        history["t"].append(step * DT)
        history["x"].append(state.x)
        history["y"].append(state.y)
        history["theta"].append(state.theta)
        history["e"].append(e)
        history["psi_e"].append(psi_e)
        history["steer"].append(steering)
        history["vy"].append(state.vy)
        history["r"].append(state.r)

        state = update_vehicle_state(state, steering)
        if index >= len(path["x"]) - 3:
            break

    result = {key: np.asarray(value, dtype=float) for key, value in history.items()}
    steer_deg = np.rad2deg(result["steer"])
    result["case"] = case.name
    result["rms_e"] = float(np.sqrt(np.mean(result["e"] ** 2)))
    result["rms_psi_deg"] = float(np.sqrt(np.mean(np.rad2deg(result["psi_e"]) ** 2)))
    result["max_abs_steer_deg"] = float(np.max(np.abs(steer_deg)))
    result["mean_abs_steer_rate_deg_s"] = float(np.mean(np.abs(np.diff(steer_deg))) / DT) if len(steer_deg) > 1 else 0.0
    return result


def run_all_cases() -> tuple[dict[str, dict[str, np.ndarray]], list[LQRCase], dict[tuple[str, str], dict[str, np.ndarray | float | str]]]:
    paths = {
        "Polynomial + sine": build_path(path_polynomial_sine),
        "Sine": build_path(path_sine),
        "Lane change": build_path(path_lane_change),
    }
    cases = [
        LQRCase("Balanced", (30.0, 45.0, 1.0, 1.0), 1.0),
        LQRCase("Aggressive", (120.0, 140.0, 1.0, 1.0), 0.25),
        LQRCase("Smooth steering", (20.0, 25.0, 1.0, 1.0), 8.0),
        LQRCase("Lateral-error focused", (180.0, 30.0, 1.0, 1.0), 1.0),
        LQRCase("Heading-error focused", (25.0, 180.0, 1.0, 1.0), 1.0),
    ]

    results = {}
    for path_name, path in paths.items():
        for case in cases:
            results[(path_name, case.name)] = simulate_path(path, case)
    return paths, cases, results


def print_gain_table(cases: list[LQRCase]) -> None:
    print("LQR gains for x = [e, psi_e, v_y, r]^T")
    print("case                    k_e      k_psi     k_vy      k_r")
    print("--------------------  -------  --------  -------  -------")
    for case in cases:
        gain = lqr_gain(case.q_diag, case.r)[0]
        print(f"{case.name:20s} {gain[0]:8.3f} {gain[1]:9.3f} {gain[2]:8.3f} {gain[3]:8.3f}")


def print_metric_table(results: dict[tuple[str, str], dict[str, np.ndarray | float | str]], cases: list[LQRCase]) -> None:
    print("\nTracking metrics")
    print("path                 case                    rms e   rms psi   max steer   mean |steer rate|")
    print("-------------------  --------------------  ------  --------  ----------  -----------------")
    for path_name in sorted({key[0] for key in results}):
        for case in cases:
            result = results[(path_name, case.name)]
            print(
                f"{path_name:19s} {case.name:20s}"
                f" {result['rms_e']:7.3f}"
                f" {result['rms_psi_deg']:9.3f}"
                f" {result['max_abs_steer_deg']:11.3f}"
                f" {result['mean_abs_steer_rate_deg_s']:18.3f}"
            )


def make_plots(
    paths: dict[str, dict[str, np.ndarray]],
    cases: list[LQRCase],
    results: dict[tuple[str, str], dict[str, np.ndarray | float | str]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (path_name, path) in zip(axes, paths.items()):
        ax.plot(path["x"], path["y"], "k--", linewidth=2, label="reference")
        for case in cases:
            result = results[(path_name, case.name)]
            ax.plot(result["x"], result["y"], linewidth=1.6, label=case.name)
        ax.set_title(path_name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)

    lane_path = "Lane change"
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for case in cases:
        result = results[(lane_path, case.name)]
        axes2[0].plot(result["t"], result["e"], label=case.name)
        axes2[1].plot(result["t"], np.rad2deg(result["steer"]), label=case.name)
    axes2[0].set_ylabel("cross-track error [m]")
    axes2[1].set_ylabel("steering [deg]")
    axes2[1].set_xlabel("time [s]")
    for ax in axes2:
        ax.grid(True, alpha=0.3)
    axes2[0].legend(loc="best", fontsize=8)
    plt.show()


if __name__ == "__main__":
    all_paths, all_cases, all_results = run_all_cases()
    print_gain_table(all_cases)
    print_metric_table(all_results, all_cases)
    make_plots(all_paths, all_cases, all_results)
