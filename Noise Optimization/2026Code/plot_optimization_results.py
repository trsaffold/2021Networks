#!/usr/bin/env python3
"""
Create plots from Kuramoto noise optimization JSON results.

The script reads every timestamped JSON file under RESULTS_DIR and writes:
  - <run>_network.png
  - <run>_covariance.png

Edit RESULTS_DIR below if you move the optimization outputs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "optimization_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(RESULTS_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def compact_vector(values: np.ndarray, max_items: int = 8) -> str:
    if values.size <= max_items:
        return "[" + ", ".join(f"{x:.3g}" for x in values) + "]"
    head_count = max_items // 2
    tail_count = max_items - head_count
    head = ", ".join(f"{x:.3g}" for x in values[:head_count])
    tail = ", ".join(f"{x:.3g}" for x in values[-tail_count:])
    return f"[{head}, ..., {tail}]"


def result_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    return sorted(
        path
        for path in results_dir.rglob("*.json")
        if path.name != "latest.json"
        and not path.name.startswith(".")
        and not any(part.startswith(".") for part in path.relative_to(results_dir).parts)
    )


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_label(data: dict, fallback: str) -> str:
    metadata = data.get("metadata", {})
    name = metadata.get("run_name") or fallback
    timestamp = metadata.get("timestamp")
    return f"{name} ({timestamp})" if timestamp else name


def network_summary(K: np.ndarray, omega: np.ndarray, theta: np.ndarray, data: dict) -> str:
    upper = np.triu(np.abs(K), k=1)
    weights = upper[upper > 1e-12]
    edge_count = int(weights.size)
    degree = np.count_nonzero(np.abs(K) > 1e-12, axis=1)
    metadata = data.get("metadata", {})
    pieces = [
        f"N = {K.shape[0]}",
        f"edges = {edge_count}",
        f"degree range = {degree.min()}-{degree.max()}",
    ]
    if weights.size:
        pieces.append(f"K range = {weights.min():.3g}-{weights.max():.3g}")
    if "coupling" in metadata:
        pieces.append(f"coupling = {metadata['coupling']:.3g}")
    pieces.extend(
        [
            f"omega range = {omega.min():.3g}-{omega.max():.3g}",
            f"theta_bar range = {theta.min():.3g}-{theta.max():.3g}",
            "theta_bar =",
            compact_vector(theta, max_items=6),
        ]
    )
    return "\n".join(pieces)


def draw_network_page(data: dict, source_path: Path):
    K = np.asarray(data["initial_conditions"]["K"], dtype=float)
    omega = np.asarray(data["initial_conditions"]["omega"], dtype=float)
    theta = np.asarray(data["fixed_point"]["theta_bar"], dtype=float)
    objective = float(data["optimization"]["objective"])
    status = data["optimization"]["cosmo_status"]
    N = K.shape[0]

    angles = np.linspace(0, 2 * math.pi, N, endpoint=False) + math.pi / 2
    positions = np.column_stack((np.cos(angles), np.sin(angles)))

    segments = []
    edge_weights = []
    for i in range(N):
        for j in range(i + 1, N):
            weight = K[i, j]
            if abs(weight) > 1e-12:
                segments.append([positions[i], positions[j]])
                edge_weights.append(abs(weight))

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    spec = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.25])
    ax = fig.add_subplot(spec[0, 0])
    info_ax = fig.add_subplot(spec[0, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    info_ax.axis("off")

    if edge_weights:
        edge_weights_arr = np.asarray(edge_weights)
        max_weight = edge_weights_arr.max()
        widths = 1.0 + 4.0 * edge_weights_arr / max_weight
        edges = LineCollection(
            segments,
            colors="#5f6673",
            linewidths=widths,
            alpha=0.55,
            zorder=1,
        )
        ax.add_collection(edges)

    theta_abs_max = max(float(np.max(np.abs(theta))), 1e-12)
    nodes = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=theta,
        cmap="coolwarm",
        vmin=-theta_abs_max,
        vmax=theta_abs_max,
        s=620,
        edgecolors="#20232a",
        linewidths=1.2,
        zorder=3,
    )

    for idx, (x, y) in enumerate(positions, start=1):
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=9, weight="bold", zorder=4)

    cbar = fig.colorbar(nodes, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label(r"$\theta_{\mathrm{bar}}$")

    info_ax.text(
        0.0,
        0.98,
        network_summary(K, omega, theta, data),
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        transform=info_ax.transAxes,
    )
    info_ax.text(
        0.0,
        0.18,
        f"objective = {objective:.6g}\nstatus = {status}\nsource = {source_path.name}",
        ha="left",
        va="bottom",
        fontsize=10,
        linespacing=1.35,
        transform=info_ax.transAxes,
    )
    ax.set_title(f"Network and Locked Phases\n{run_label(data, source_path.stem)}", fontsize=14)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    return fig


def draw_covariance_page(data: dict, source_path: Path):
    C = np.asarray(data["optimization"]["C_opt"], dtype=float)
    theta = np.asarray(data["fixed_point"]["theta_bar"], dtype=float)
    K = np.asarray(data["initial_conditions"]["K"], dtype=float)
    residual = float(data["optimization"]["max_lyapunov_residual_inf_norm"])
    objective = float(data["optimization"]["objective"])
    eig_min = float(np.min(np.asarray(data["optimization"]["C_eigenvalues"], dtype=float)))

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    spec = fig.add_gridspec(1, 2, width_ratios=[16, 1.2])
    ax = fig.add_subplot(spec[0, 0])
    theta_ax = fig.add_subplot(spec[0, 1])

    vmax = max(float(np.max(np.abs(C))), 1e-12)
    im = ax.imshow(C, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Optimal Covariance Matrix\n{run_label(data, source_path.stem)}", fontsize=14)
    ax.set_xlabel("Oscillator j")
    ax.set_ylabel("Oscillator i")
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_xticklabels(np.arange(1, C.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, C.shape[0] + 1))
    ax.tick_params(labelsize=8)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C.shape[0] <= 16:
                color = "white" if abs(C[i, j]) > 0.55 * vmax else "#20232a"
                ax.text(j, i, f"{C[i, j]:.2g}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(r"$C_{ij}^{opt}$")

    theta_abs_max = max(float(np.max(np.abs(theta))), 1e-12)
    theta_im = theta_ax.imshow(theta[:, None], cmap="coolwarm", vmin=-theta_abs_max, vmax=theta_abs_max, aspect="auto")
    theta_ax.set_title(r"$\theta_{\mathrm{bar}}$", fontsize=10)
    theta_ax.set_xticks([])
    theta_ax.set_yticks(np.arange(theta.size))
    theta_ax.set_yticklabels(np.arange(1, theta.size + 1))
    theta_ax.tick_params(labelsize=7)
    fig.colorbar(theta_im, ax=theta_ax, shrink=0.82, pad=0.18)

    upper = np.triu(np.abs(K), k=1)
    weights = upper[upper > 1e-12]
    k_summary = "none" if weights.size == 0 else f"{weights.min():.3g}-{weights.max():.3g}"
    ax.text(
        0.0,
        -0.15,
        f"objective = {objective:.6g}    min eig(C) = {eig_min:.3g}    max Lyapunov residual = {residual:.3g}    K range = {k_summary}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    return fig


def save_plots_for_result(path: Path, output_dir: Path) -> list[Path]:
    data = load_result(path)
    stem = path.stem

    network_fig = draw_network_page(data, path)
    covariance_fig = draw_covariance_page(data, path)

    network_path = output_dir / f"{stem}_network.png"
    covariance_path = output_dir / f"{stem}_covariance.png"

    network_fig.savefig(network_path, dpi=220)
    covariance_fig.savefig(covariance_path, dpi=220)

    plt.close(network_fig)
    plt.close(covariance_fig)
    return [network_path, covariance_path]


def main() -> None:
    files = result_files(RESULTS_DIR)
    if not files:
        raise SystemExit(f"No timestamped JSON result files found in {RESULTS_DIR}")

    written = []
    for path in files:
        written.extend(save_plots_for_result(path, path.parent))

    print(f"Processed {len(files)} result file(s).")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
