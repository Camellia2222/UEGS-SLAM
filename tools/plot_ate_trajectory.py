"""Plot KITTI trajectories in an APE-style panel."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


AXIS_NAMES = ("x", "y", "z")


def load_kitti_positions(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] == 12:
        poses = data.reshape(-1, 3, 4)
    elif data.shape[1] == 16:
        poses = data.reshape(-1, 4, 4)[:, :3, :]
    else:
        raise ValueError(
            f"{path} must contain 12 (3x4) or 16 (4x4) values per line; "
            f"got {data.shape[1]}."
        )
    positions = poses[:, :3, 3]
    if not np.isfinite(positions).all():
        raise ValueError(f"{path} contains NaN or Inf positions.")
    return positions


def umeyama_align(source: np.ndarray, target: np.ndarray, with_scale: bool):
    """Return source aligned to target and the applied Sim(3)/SE(3)."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt

    if with_scale:
        variance = np.mean(np.sum(source_centered**2, axis=1))
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("Estimated trajectory has zero spatial variance.")
        scale = np.sum(singular_values * np.diag(correction)) / variance
    else:
        scale = 1.0

    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation, translation


def parse_estimate(spec: str):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).stem.replace("pose_file_est_", "")
    if not label:
        raise ValueError(f"Invalid estimate label in: {spec}")
    return label, Path(path)


def choose_plane(gt: np.ndarray, plane: str):
    if plane != "auto":
        lookup = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        return lookup[plane]
    spread = np.ptp(gt, axis=0)
    axes = np.argsort(spread)[-2:]
    return tuple(sorted(int(axis) for axis in axes))


def colored_trajectory(ax, positions, errors, axes, cmap, norm):
    points = positions[:, axes].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_errors = 0.5 * (errors[:-1] + errors[1:])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=1.45,
        capstyle="round",
        zorder=3,
    )
    collection.set_array(segment_errors)
    ax.add_collection(collection)
    return collection


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot KITTI trajectories with translation APE mapped to the "
            "Figure 4 color style. Basic usage: "
            "plot_ate_trajectory.py GT.txt EST.txt OUTPUT.png"
        )
    )
    parser.add_argument("gt", type=Path, help="Ground-truth KITTI pose file.")
    parser.add_argument("est", type=Path, help="Estimated KITTI pose file.")
    parser.add_argument("output", type=Path, help="Output .png or .pdf path.")
    parser.add_argument(
        "--label",
        default="ours",
        help="Estimate label used in terminal statistics.",
    )
    parser.add_argument(
        "--extra-est",
        action="append",
        metavar="LABEL=PATH",
        help="Optional additional estimate. Repeat to create comparison panels.",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Optional scene name used only in terminal output.",
    )
    parser.add_argument(
        "--align",
        choices=("sim3", "se3", "none"),
        default="sim3",
        help="sim3 matches evo --align --correct_scale; se3 aligns without scale.",
    )
    parser.add_argument("--plane", choices=("auto", "xy", "xz", "yz"), default="yz")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--error-min", type=float, default=None)
    parser.add_argument("--error-max", type=float, default=None)
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster DPI. The default produces an 800x600 PNG for one estimate.",
    )
    args = parser.parse_args()

    gt_all = load_kitti_positions(args.gt)
    estimates = []
    estimate_specs = [f"{args.label}={args.est}"] + (args.extra_est or [])
    for spec in estimate_specs:
        label, path = parse_estimate(spec)
        est_all = load_kitti_positions(path)
        count = min(len(gt_all), len(est_all))
        if count < 2:
            raise ValueError(f"Need at least two matching poses for {label}.")
        if len(gt_all) != len(est_all):
            print(
                f"[warning] {label}: GT has {len(gt_all)} poses and EST has "
                f"{len(est_all)}; using the first {count}."
            )
        gt = gt_all[:count]
        est = est_all[:count]
        if args.align == "sim3":
            aligned, scale, _, _ = umeyama_align(est, gt, with_scale=True)
        elif args.align == "se3":
            aligned, scale, _, _ = umeyama_align(est, gt, with_scale=False)
        else:
            aligned, scale = est.copy(), 1.0
        errors = np.linalg.norm(aligned - gt, axis=1)
        estimates.append((label, gt, aligned, errors, scale))

    all_errors = np.concatenate([item[3] for item in estimates])
    error_min = float(all_errors.min()) if args.error_min is None else args.error_min
    error_max = float(all_errors.max()) if args.error_max is None else args.error_max
    if error_max <= error_min:
        error_max = error_min + np.finfo(np.float64).eps
    norm = Normalize(vmin=error_min, vmax=error_max)
    cmap = plt.get_cmap(args.cmap)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    width = 4.0 * len(estimates)
    if len(estimates) == 1:
        # Match the 800x600 single-trajectory EVO panels.
        fig = plt.figure(figsize=(4.0, 3.0), facecolor="white")
        subplot_axes = [fig.add_axes([0.143, 0.215, 0.665, 0.57])]
    else:
        fig, subplot_grid = plt.subplots(
            1,
            len(estimates),
            figsize=(width, 3.0),
            squeeze=False,
            facecolor="white",
        )
        subplot_axes = subplot_grid[0]

    for ax, (label, gt, aligned, errors, scale) in zip(subplot_axes, estimates):
        plane_axes = choose_plane(gt, args.plane)
        ax.plot(
            gt[:, plane_axes[0]],
            gt[:, plane_axes[1]],
            linestyle="--",
            color="0.48",
            linewidth=1.0,
            label="reference",
            zorder=2,
        )
        collection = colored_trajectory(ax, aligned, errors, plane_axes, cmap, norm)
        start_color = cmap(norm(errors[0]))
        end_color = cmap(norm(errors[-1]))
        ax.scatter(
            aligned[0, plane_axes[0]], aligned[0, plane_axes[1]],
            marker="o", s=18, c=[start_color], edgecolors="black",
            linewidths=0.55, zorder=4,
        )
        ax.scatter(
            aligned[-1, plane_axes[0]], aligned[-1, plane_axes[1]],
            marker="x", s=26, c=[end_color], linewidths=0.9, zorder=4,
        )
        rmse = float(np.sqrt(np.mean(errors**2)))
        alignment_name = {
            "sim3": "Sim(3) Umeyama alignment",
            "se3": "SE(3) Umeyama alignment",
            "none": "no alignment",
        }[args.align]
        ax.set_title(
            "APE w.r.t. translation part (m)\n"
            f"(with {alignment_name})",
            pad=4,
        )
        ax.set_xlabel(f"{AXIS_NAMES[plane_axes[0]]} (m)")
        ax.set_ylabel(f"{AXIS_NAMES[plane_axes[1]]} (m)")
        ax.grid(True, color="0.78", linewidth=0.55, alpha=0.55)
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.82,
            borderpad=0.25,
            handlelength=1.8,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.04)
        if len(estimates) == 1:
            colorbar_axis = fig.add_axes([0.848, 0.05, 0.035, 0.90])
            colorbar = fig.colorbar(collection, cax=colorbar_axis)
        else:
            colorbar = fig.colorbar(collection, ax=ax, fraction=0.052, pad=0.025)
        colorbar.set_ticks(np.linspace(error_min, error_max, 3))
        colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
        colorbar.outline.set_linewidth(0.45)
        print(
            f"{args.scene + ': ' if args.scene else ''}{label}: "
            f"poses={len(errors)}, RMSE={rmse:.6f} m, "
            f"mean={errors.mean():.6f} m, median={np.median(errors):.6f} m, "
            f"max={errors.max():.6f} m, scale={scale:.6f}"
        )

    if len(estimates) > 1:
        fig.subplots_adjust(
            left=0.145 / len(estimates),
            right=0.925,
            bottom=0.155,
            top=0.81,
            wspace=0.34,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=args.dpi,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved trajectory figure to: {args.output}")


if __name__ == "__main__":
    main()
