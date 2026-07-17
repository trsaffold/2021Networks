"""
Animate a network governed by the swing equations.

Dependencies:
    pip install numpy scipy matplotlib networkx pillow
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp


def next_run_path(results_root: str = "results") -> Path:
    """
    Return results/runK.gif, where K is one greater than the
    largest existing run number.
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    run_numbers = []

    for path in root.iterdir():
        if not path.is_file():
            continue

        if (
            path.suffix.lower() == ".gif"
            and path.stem.startswith("run")
            and path.stem[3:].isdigit()
        ):
            run_numbers.append(int(path.stem[3:]))

    next_number = max(run_numbers, default=0) + 1
    return root / f"run{next_number}.gif"


def adjacency_from_edges(
    n_oscillators: int,
    edges: list[tuple[int, int, float]],
) -> np.ndarray:
    """
    Construct a symmetric weighted adjacency matrix.

    Each edge is:
        (node_i, node_j, coupling_weight)
    """
    adjacency = np.zeros((n_oscillators, n_oscillators), dtype=float)

    for i, j, weight in edges:
        adjacency[i, j] = weight
        adjacency[j, i] = weight

    return adjacency


def swing_rhs(
    t: float,
    state: np.ndarray,
    power_inputs: np.ndarray,
    inertia: np.ndarray,
    damping: np.ndarray,
    coupling_matrix: np.ndarray,
) -> np.ndarray:
    """
    Swing equations:

        dtheta_i/dt = frequency_i

        M_i d(frequency_i)/dt
            = P_i
              - D_i frequency_i
              + sum_j K_ij sin(theta_j - theta_i)

    The state vector is:

        state = [theta_1, ..., theta_N,
                 frequency_1, ..., frequency_N]
    """
    n = len(power_inputs)

    theta = state[:n]
    frequency = state[n:]

    phase_difference = theta[np.newaxis, :] - theta[:, np.newaxis]

    electrical_coupling = (
        coupling_matrix * np.sin(phase_difference)
    ).sum(axis=1)

    angular_acceleration = (
        power_inputs
        - damping * frequency
        + electrical_coupling
    ) / inertia

    return np.concatenate([frequency, angular_acceleration])


def simulate_swing_network(
    coupling_matrix: np.ndarray,
    power_inputs: np.ndarray,
    inertia: np.ndarray,
    damping: np.ndarray,
    initial_phases: np.ndarray,
    initial_frequencies: np.ndarray,
    duration: float = 30.0,
    fps: int = 20,
    output_path: str = "results/run1.gif",
    layout_seed: int = 4,
    reference_frequency: float = 2.0,
    ) -> None:
    coupling_matrix = np.asarray(coupling_matrix, dtype=float)
    power_inputs = np.asarray(power_inputs, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    damping = np.asarray(damping, dtype=float)
    initial_phases = np.asarray(initial_phases, dtype=float)
    initial_frequencies = np.asarray(initial_frequencies, dtype=float)

    n = len(power_inputs)

    expected_vector_shape = (n,)

    for name, array in [
        ("inertia", inertia),
        ("damping", damping),
        ("initial_phases", initial_phases),
        ("initial_frequencies", initial_frequencies),
    ]:
        if array.shape != expected_vector_shape:
            raise ValueError(
                f"{name} must have shape {expected_vector_shape}."
            )

    if coupling_matrix.shape != (n, n):
        raise ValueError(
            f"coupling_matrix must have shape ({n}, {n})."
        )

    if not np.allclose(coupling_matrix, coupling_matrix.T):
        raise ValueError(
            "coupling_matrix must be symmetric for an undirected network."
        )

    if np.any(inertia <= 0):
        raise ValueError("Every inertia value must be positive.")

    if np.any(damping < 0):
        raise ValueError("Damping values must be nonnegative.")

    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive.")

    # In a centered grid model, total net power should generally sum to zero.
    if not np.isclose(power_inputs.sum(), 0.0):
        print(
            "Warning: power_inputs do not sum to zero. "
            "The network may develop a nonzero common acceleration or frequency."
        )

    n_frames = int(duration * fps) + 1
    times = np.linspace(0.0, duration, n_frames)

    initial_state = np.concatenate(
        [initial_phases, initial_frequencies]
    )

    solution = solve_ivp(
        fun=swing_rhs,
        t_span=(0.0, duration),
        y0=initial_state,
        t_eval=times,
        args=(
            power_inputs,
            inertia,
            damping,
            coupling_matrix,
        ),
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    phases = solution.y[:n]
    frequency_deviations = solution.y[n:]

    physical_phases = (
        phases
        + reference_frequency * times[np.newaxis, :]
    )

    physical_frequencies = (
        frequency_deviations
        + reference_frequency
    )

    # Kuramoto order parameter.
    order_parameter = np.abs(
        np.mean(np.exp(1j * physical_phases), axis=0)
    ) ** 2

    graph = nx.from_numpy_array(coupling_matrix)

    positions_dict = nx.spring_layout(
        graph,
        seed=layout_seed,
        weight="weight",
    )

    positions = np.array(
        [positions_dict[i] for i in range(n)]
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.5, 1.0]},
    )

    network_ax, frequency_ax, order_ax = axes

    figure.suptitle(
        "Swing-equation oscillator network",
        fontsize=14,
    )

    # --------------------------------------------------
    # Network panel
    # --------------------------------------------------
    network_ax.set_title("Rotor/electrical phases")
    network_ax.set_aspect("equal")
    network_ax.axis("off")

    edge_segments = []
    edge_widths = []

    nonzero_weights = coupling_matrix[
        coupling_matrix > 0
    ]

    max_weight = (
        nonzero_weights.max()
        if nonzero_weights.size > 0
        else 1.0
    )

    for i, j in graph.edges():
        edge_segments.append(
            [positions[i], positions[j]]
        )

        normalized_weight = (
            coupling_matrix[i, j] / max_weight
        )

        edge_widths.append(
            1.0 + 3.0 * normalized_weight
        )

    edge_collection = LineCollection(
        edge_segments,
        linewidths=edge_widths,
        alpha=0.5,
        zorder=1,
    )

    network_ax.add_collection(edge_collection)

    padding = 0.35

    network_ax.set_xlim(
        positions[:, 0].min() - padding,
        positions[:, 0].max() + padding,
    )

    network_ax.set_ylim(
        positions[:, 1].min() - padding,
        positions[:, 1].max() + padding,
    )

    node_scatter = network_ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.mod(physical_phases[:, 0], 2 * np.pi),
        cmap="hsv",
        vmin=0.0,
        vmax=2 * np.pi,
        s=900,
        edgecolors="black",
        linewidths=1.5,
        zorder=2,
    )

    for i, (x, y) in enumerate(positions):
        network_ax.text(
            x,
            y,
            str(i),
            ha="center",
            va="center",
            fontsize=10,
            zorder=4,
        )

    hand_length = 0.12
    phase_hands = []

    for i, (x, y) in enumerate(positions):
        phase = physical_phases[i, 0]

        hand, = network_ax.plot(
            [
                x,
                x + hand_length * np.cos(phase),
            ],
            [
                y,
                y + hand_length * np.sin(phase),
            ],
            linewidth=2.5,
            zorder=3,
        )

        phase_hands.append(hand)

    time_text = network_ax.text(
        0.02,
        0.98,
        "",
        transform=network_ax.transAxes,
        ha="left",
        va="top",
    )

    # --------------------------------------------------
    # Frequency panel
    # --------------------------------------------------
    frequency_ax.set_title("Frequency deviations")
    frequency_ax.set_xlabel("Time")
    frequency_ax.set_ylabel(
    r"Physical frequency $\omega_{\rm ref}+\dot{\theta}_i$"
    )
    frequency_ax.set_xlim(0.0, duration)
    frequency_ax.grid(alpha=0.25)

    frequency_min = physical_frequencies.min()
    frequency_max = physical_frequencies.max()

    frequency_padding = max(
        0.05,
        0.1 * (frequency_max - frequency_min),
    )

    frequency_ax.set_ylim(
        frequency_min - frequency_padding,
        frequency_max + frequency_padding,
    )

    frequency_lines = []

    for i in range(n):
        line, = frequency_ax.plot(
            [],
            [],
            linewidth=2,
            label=(
                f"Node {i}: "
                f"P={power_inputs[i]:.2f}, "
                f"M={inertia[i]:.2f}"
            ),
        )

        frequency_lines.append(line)

    frequency_ax.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
        alpha=0.6,
    )

    frequency_ax.legend(
        loc="best",
        fontsize=7,
    )

    # --------------------------------------------------
    # Order-parameter panel
    # --------------------------------------------------
    order_ax.set_title(r"Synchronization $R^2$")
    order_ax.set_xlabel("Time")
    order_ax.set_ylabel(r"$R^2$")
    order_ax.set_xlim(0.0, duration)
    order_ax.set_ylim(0.0, 1.05)
    order_ax.grid(alpha=0.25)

    order_line, = order_ax.plot(
        [],
        [],
        linewidth=2.5,
    )

    order_text = order_ax.text(
        0.05,
        0.95,
        "",
        transform=order_ax.transAxes,
        ha="left",
        va="top",
    )

    def update(frame: int):
        current_phases = physical_phases[:, frame]

        node_scatter.set_array(
            np.mod(current_phases, 2 * np.pi)
        )

        for i, hand in enumerate(phase_hands):
            x, y = positions[i]
            phase = current_phases[i]

            hand.set_data(
                [
                    x,
                    x + hand_length * np.cos(phase),
                ],
                [
                    y,
                    y + hand_length * np.sin(phase),
                ],
            )

        for i, line in enumerate(frequency_lines):
            line.set_data(
                times[: frame + 1],
                physical_frequencies[i, : frame + 1],
            )

        order_line.set_data(
            times[: frame + 1],
            order_parameter[: frame + 1],
        )

        time_text.set_text(
            f"t = {times[frame]:.2f}"
        )

        order_text.set_text(
            rf"$R^2={order_parameter[frame]:.3f}$"
        )

        return [
            node_scatter,
            time_text,
            order_line,
            order_text,
            *phase_hands,
            *frequency_lines,
        ]

    animation = FuncAnimation(
        figure,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    animation.save(
        output,
        writer=PillowWriter(fps=fps),
        dpi=110,
    )

    plt.close(figure)

    print(f"Saved animation to: {output.resolve()}")


if __name__ == "__main__":
    # Each tuple is:
    #     (node i, node j, coupling strength K_ij)
    edge_list = [
        (0, 1, 2.0),
        (1, 2, 1.5),
        (2, 3, 2.0),
        (3, 4, 1.5),
        (4, 0, 2.0),
        (0, 2, 0.8),
        (1, 3, 0.8),
    ]

    n_nodes = 5

    coupling_matrix = adjacency_from_edges(
        n_oscillators=n_nodes,
        edges=edge_list,
    )

    # Positive values represent net generation.
    # Negative values represent net load.
    #
    # For a centered grid, these should sum to zero.
    power_inputs = np.array(
        [1.0, 0.6, -0.4, -0.7, -0.5]
    )

    # Generator or virtual inertia.
    inertia = np.array(
        [2.0, 1.8, 1.5, 1.7, 2.2]
    )

    # Damping or frequency-response strength.
    damping = np.array(
        [1.0, 1.0, 1.2, 1.1, 1.0]
    )

    # Initial phase angles in radians.
    initial_phases = np.array(
        [0.0, 0.7, 1.8, 3.2, 4.8]
    )

    # Initial frequency deviations.

    # Non-center frame (powers add to 0):
    initial_frequencies = np.array(
        [0.0, -1.3, 1.7, 0.9, -0.3]
    )

    gif_path = next_run_path("swing_results")

    simulate_swing_network(
        coupling_matrix=coupling_matrix,
        power_inputs=power_inputs,
        inertia=inertia,
        damping=damping,
        initial_phases=initial_phases,
        initial_frequencies=initial_frequencies,
        duration=40.0,
        fps=20,
        output_path=str(gif_path),
        reference_frequency=3.0
    )