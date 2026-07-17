"""
Animate a small Kuramoto oscillator network.

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

        name = path.stem  # e.g. "run3"

        if name.startswith("run") and name[3:].isdigit() and path.suffix == ".gif":
            run_numbers.append(int(name[3:]))

    next_number = max(run_numbers, default=0) + 1
    return root / f"run{next_number}.gif"


def kuramoto_rhs(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    adjacency: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """
    Kuramoto equations:

        dtheta_i/dt = omega_i
                      + coupling * sum_j A_ij sin(theta_j - theta_i)

    The normalization by weighted degree keeps coupling magnitudes comparable
    across nodes with different numbers of neighbors.
    """
    phase_difference = theta[np.newaxis, :] - theta[:, np.newaxis]
    interaction = adjacency * np.sin(phase_difference)

    weighted_degree = adjacency.sum(axis=1)
    weighted_degree = np.where(weighted_degree > 0, weighted_degree, 1.0)

    return omega + coupling * interaction.sum(axis=1) / weighted_degree


def adjacency_from_edges(
    n_oscillators: int,
    edges: list[tuple[int, int, float]],
) -> np.ndarray:
    """
    Construct a symmetric weighted adjacency matrix.

    Each edge is (node_i, node_j, weight).
    """
    adjacency = np.zeros((n_oscillators, n_oscillators), dtype=float)

    for i, j, weight in edges:
        adjacency[i, j] = weight
        adjacency[j, i] = weight

    return adjacency


def simulate_network(
    adjacency: np.ndarray,
    natural_frequencies: np.ndarray,
    initial_phases: np.ndarray,
    coupling: float = 2.0,
    duration: float = 20.0,
    fps: int = 20,
    output_path: str = "kuramoto_network.gif",
    layout_seed: int = 4,
) -> None:
    adjacency = np.asarray(adjacency, dtype=float)
    omega = np.asarray(natural_frequencies, dtype=float)
    theta_0 = np.asarray(initial_phases, dtype=float)

    n = len(omega)

    if adjacency.shape != (n, n):
        raise ValueError(f"adjacency must have shape ({n}, {n}).")

    if theta_0.shape != (n,):
        raise ValueError(f"initial_phases must have shape ({n},).")

    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("This script expects an undirected symmetric adjacency matrix.")

    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive.")

    n_frames = int(duration * fps) + 1
    times = np.linspace(0.0, duration, n_frames)

    solution = solve_ivp(
        fun=kuramoto_rhs,
        t_span=(0.0, duration),
        y0=theta_0,
        t_eval=times,
        args=(omega, adjacency, coupling),
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    phases = solution.y

    # Calculate instantaneous frequencies from the differential equation itself.
    instantaneous_frequencies = np.column_stack(
        [
            kuramoto_rhs(
                t=times[k],
                theta=phases[:, k],
                omega=omega,
                adjacency=adjacency,
                coupling=coupling,
            )
            for k in range(n_frames)
        ]
    )

    graph = nx.from_numpy_array(adjacency)
    positions_dict = nx.spring_layout(graph, seed=layout_seed, weight="weight")
    positions = np.array([positions_dict[i] for i in range(n)])

    figure, (network_ax, frequency_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1, 1.5]},
    )

    figure.suptitle(
        f"Kuramoto network: coupling K = {coupling:.2f}",
        fontsize=14,
    )

    # -----------------------
    # Network panel
    # -----------------------
    network_ax.set_title("Oscillator phases")
    network_ax.set_aspect("equal")
    network_ax.axis("off")

    edge_segments = []
    edge_widths = []

    for i, j in graph.edges():
        edge_segments.append([positions[i], positions[j]])
        edge_widths.append(1.0 + 1.5 * adjacency[i, j])

    edge_collection = LineCollection(
        edge_segments,
        linewidths=edge_widths,
        alpha=0.5,
        zorder=1,
    )
    network_ax.add_collection(edge_collection)

    padding = 0.35
    network_ax.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    network_ax.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)

    node_scatter = network_ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.mod(phases[:, 0], 2 * np.pi),
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

    # Small clock-hand line for each oscillator.
    hand_length = 0.12
    phase_hands = []

    for i, (x, y) in enumerate(positions):
        phase = phases[i, 0]
        hand, = network_ax.plot(
            [x, x + hand_length * np.cos(phase)],
            [y, y + hand_length * np.sin(phase)],
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

    # -----------------------
    # Frequency panel
    # -----------------------
    frequency_ax.set_title("Instantaneous oscillator frequencies")
    frequency_ax.set_xlabel("Time")
    frequency_ax.set_ylabel(r"$d\theta_i/dt$")
    frequency_ax.set_xlim(0.0, duration)

    frequency_min = instantaneous_frequencies.min()
    frequency_max = instantaneous_frequencies.max()
    frequency_padding = max(0.1, 0.1 * (frequency_max - frequency_min))

    frequency_ax.set_ylim(
        frequency_min - frequency_padding,
        frequency_max + frequency_padding,
    )
    frequency_ax.grid(alpha=0.25)

    frequency_lines = []

    for i in range(n):
        line, = frequency_ax.plot(
            [],
            [],
            linewidth=2,
            label=rf"Oscillator {i}, $\omega_i={omega[i]:.2f}$",
        )
        frequency_lines.append(line)

    frequency_ax.legend(loc="best", fontsize=8)

    def update(frame: int):
        current_phases = phases[:, frame]

        node_scatter.set_array(np.mod(current_phases, 2 * np.pi))

        for i, hand in enumerate(phase_hands):
            x, y = positions[i]
            phase = current_phases[i]

            hand.set_data(
                [x, x + hand_length * np.cos(phase)],
                [y, y + hand_length * np.sin(phase)],
            )

        for i, line in enumerate(frequency_lines):
            line.set_data(
                times[: frame + 1],
                instantaneous_frequencies[i, : frame + 1],
            )

        time_text.set_text(f"t = {times[frame]:.2f}")

        return [node_scatter, time_text, *phase_hands, *frequency_lines]

    animation = FuncAnimation(
        figure,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    animation.save(
        output,
        writer=PillowWriter(fps=fps),
        dpi=110,
    )

    plt.close(figure)
    print(f"Saved animation to: {output.resolve()}")


if __name__ == "__main__":
    edge_list = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 0, 1.0),
        (0, 2, 0.5),
        (1, 3, 0.5),
    ]

    adjacency_matrix = adjacency_from_edges(
        n_oscillators=5,
        edges=edge_list,
    )

    omega = np.array([0.30, 0.92, 1.00, 1.6, 1.20])
    initial_theta = np.array([0.0, 1.7, 3.5, 5.0, 2.5])

    gif_path = next_run_path("KM_results")

    simulate_network(
        adjacency=adjacency_matrix,
        natural_frequencies=omega,
        initial_phases=initial_theta,
        coupling=1.0,
        duration=30.0,
        fps=20,
        output_path=str(gif_path),
    )