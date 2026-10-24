"""Persistent loss logging and plotting for full-sequence experiments."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np


class LossDiagnostics:
    def __init__(
        self,
        output_root: str,
        enabled: bool = True,
        log_every: int = 10,
        plot_every_windows: int = 5,
    ):
        self.enabled = bool(enabled)
        self.log_every = max(1, int(log_every))
        self.plot_every_windows = max(1, int(plot_every_windows))
        self._optimization_id = 0
        self._completed_local_windows = 0

        base_root = Path(output_root)
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.root = base_root / run_name
        if not self.enabled:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        (base_root / "LATEST.txt").write_text(run_name + "\n", encoding="utf-8")
        (self.root / "README.txt").write_text(
            "loss_history.csv contains detached BA losses.\n"
            "maintenance_history.csv contains Gaussian maintenance counts.\n"
            "loss_convergence.png summarizes convergence within local BA windows.\n"
            "loss_full_sequence.png shows terminal loss and reduction over frame IDs.\n"
            "gaussian_maintenance_full_sequence.png shows maintenance counts over frame IDs.\n"
            "gaussian_maintenance_normalized.png shows map-size-normalized filtering ratios.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _scalar(value):
        if value is None:
            return float("nan")
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value)
        return float(array.reshape(-1)[0]) if array.size else float("nan")

    def start_optimization(self, stage: str, frame_id: int):
        if not self.enabled:
            return -1
        optimization_id = self._optimization_id
        self._optimization_id += 1
        return optimization_id

    def record(
        self,
        optimization_id: int,
        stage: str,
        frame_id: int,
        local_step: int,
        max_steps: int,
        **losses,
    ):
        if not self.enabled:
            return
        endpoint = int(local_step) in (0, max(0, int(max_steps) - 1))
        if not endpoint and int(local_step) % self.log_every != 0:
            return

        path = self.root / "loss_history.csv"
        row = {
            "optimization_id": int(optimization_id),
            "stage": str(stage),
            "frame_id": int(frame_id),
            "local_step": int(local_step),
            "max_steps": int(max_steps),
            "progress": int(local_step) / max(1, int(max_steps) - 1),
        }
        row.update({name: self._scalar(value) for name, value in losses.items()})
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

        is_last_step = int(local_step) == max(0, int(max_steps) - 1)
        if is_last_step and stage == "local_BA":
            self._completed_local_windows += 1
            if self._completed_local_windows % self.plot_every_windows == 0:
                self.render()

    def record_maintenance(
        self,
        stage: str,
        frame_id: int,
        local_step: int,
        total_gaussians: int,
        eligible_gaussians: int,
        joint_candidates: int,
        persistent_candidates: int,
        attenuated_gaussians: int,
    ):
        if not self.enabled:
            return
        path = self.root / "maintenance_history.csv"
        row = {
            "stage": str(stage),
            "frame_id": int(frame_id),
            "local_step": int(local_step),
            "total_gaussians": int(total_gaussians),
            "eligible_gaussians": int(eligible_gaussians),
            "joint_candidates": int(joint_candidates),
            "persistent_candidates": int(persistent_candidates),
            "attenuated_gaussians": int(attenuated_gaussians),
        }
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

    @staticmethod
    def _read_rows(path):
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _values(rows, key):
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                values.append(np.nan)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _plt():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _local_groups(self, rows):
        groups = {}
        for row in rows:
            if row.get("stage") == "local_BA":
                groups.setdefault(row["optimization_id"], []).append(row)
        return {
            key: sorted(group, key=lambda row: float(row["progress"]))
            for key, group in groups.items()
            if len(group) >= 2
        }

    def _plot_convergence(self, rows):
        groups = {key: group for key, group in self._local_groups(rows).items() if len(group) >= 3}
        if not groups:
            return
        grid = np.linspace(0.0, 1.0, 101)

        def aggregate(key):
            curves = []
            for group in groups.values():
                progress = self._values(group, "progress")
                values = self._values(group, key)
                valid = np.isfinite(progress) & np.isfinite(values)
                if valid.sum() < 3 or abs(values[valid][0]) < 1e-12:
                    continue
                normalized = values[valid] / values[valid][0]
                curves.append(np.interp(grid, progress[valid], normalized))
            if not curves:
                return None
            stacked = np.stack(curves)
            return (
                np.nanmedian(stacked, axis=0),
                np.nanpercentile(stacked, 25, axis=0),
                np.nanpercentile(stacked, 75, axis=0),
                stacked.shape[0],
            )

        plt = self._plt()
        fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
        plotted = False
        for key, label, color in (
            ("total_loss", "total loss", "#1f77b4"),
            ("event_loss", "event loss", "#d95f02"),
        ):
            result = aggregate(key)
            if result is None:
                continue
            median, lower, upper, count = result
            ax.plot(grid, median, color=color, linewidth=2, label=f"{label} (n={count})")
            ax.fill_between(grid, lower, upper, color=color, alpha=0.18)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel("Normalized local BA progress")
        ax.set_ylabel("Loss normalized by window start")
        ax.set_title("Optimization convergence across local BA windows")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(self.root / "loss_convergence.png", dpi=220)
        plt.close(fig)

    def _plot_full_sequence(self, rows):
        groups = self._local_groups(rows)
        if not groups:
            return

        summaries = []
        for group in groups.values():
            start_total = self._values(group[:1], "total_loss")[0]
            end_total = self._values(group[-1:], "total_loss")[0]
            end_event = self._values(group[-1:], "event_loss")[0]
            summaries.append(
                (
                    float(group[-1]["frame_id"]),
                    end_total,
                    end_event,
                    end_total / start_total if abs(start_total) > 1e-12 else np.nan,
                )
            )
        summaries.sort(key=lambda item: item[0])
        data = np.asarray(summaries, dtype=np.float64)
        frames, terminal_total, terminal_event, reduction = data.T

        plt = self._plt()
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True, constrained_layout=True)
        axes[0].plot(frames, terminal_total, color="#1f77b4", linewidth=1.7, label="terminal total loss")
        axes[0].plot(frames, terminal_event, color="#d95f02", linewidth=1.5, label="terminal event loss")
        positive = np.concatenate((terminal_total, terminal_event))
        if np.all(positive[np.isfinite(positive)] > 0):
            axes[0].set_yscale("log")
        axes[0].set_ylabel("Terminal loss")
        axes[0].set_title("Full-sequence BA loss")
        axes[0].legend(fontsize=8)

        axes[1].plot(frames, reduction, color="#2a9d8f", linewidth=1.7)
        axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Frame ID")
        axes[1].set_ylabel("Final / initial total loss")
        axes[1].set_xlim(0.0, max(1.0, float(np.nanmax(frames))))
        axes[1].set_ylim(bottom=0.0)
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.savefig(self.root / "loss_full_sequence.png", dpi=220)
        plt.close(fig)

    def _plot_maintenance(self):
        rows = self._read_rows(self.root / "maintenance_history.csv")
        rows = [
            row
            for row in rows
            if row.get("stage") == "local_BA" and float(row.get("frame_id", -1)) >= 0
        ]
        if not rows:
            return
        rows.sort(key=lambda row: (float(row["frame_id"]), float(row["local_step"])))
        frames = self._values(rows, "frame_id")

        plt = self._plt()
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(
            frames,
            self._values(rows, "joint_candidates"),
            color="#4c78a8",
            linewidth=2.0,
            label="joint",
        )
        ax.plot(
            frames,
            self._values(rows, "persistent_candidates"),
            color="#f58518",
            linewidth=1.7,
            marker="o",
            markersize=3.5,
            label="persistent",
        )
        ax.plot(
            frames,
            self._values(rows, "attenuated_gaussians"),
            color="#54a24b",
            linewidth=1.5,
            marker="x",
            markersize=5.0,
            label="attenuated",
        )
        final_frame = max(1.0, float(np.nanmax(frames)))
        ax.set_xlim(0.0, final_frame)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Gaussian count")
        ax.set_title("Full-sequence Gaussian maintenance")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(self.root / "gaussian_maintenance_full_sequence.png", dpi=220)
        plt.close(fig)

    def _plot_maintenance_normalized(self):
        rows = self._read_rows(self.root / "maintenance_history.csv")
        rows = [
            row
            for row in rows
            if row.get("stage") == "local_BA" and float(row.get("frame_id", -1)) >= 0
        ]
        if not rows:
            return
        rows.sort(key=lambda row: (float(row["frame_id"]), float(row["local_step"])))
        frames = self._values(rows, "frame_id")
        eligible = np.maximum(1.0, self._values(rows, "eligible_gaussians"))
        joint = self._values(rows, "joint_candidates")
        persistent = self._values(rows, "persistent_candidates")
        attenuated = self._values(rows, "attenuated_gaussians")

        plt = self._plt()
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(frames, joint / eligible, color="#4c78a8", linewidth=2.0, label="joint / eligible")
        ax.plot(
            frames,
            persistent / np.maximum(1.0, joint),
            color="#f58518",
            linewidth=1.7,
            label="persistent / joint",
        )
        ax.plot(
            frames,
            attenuated / np.maximum(1.0, persistent),
            color="#54a24b",
            linewidth=1.7,
            label="attenuated / persistent",
        )
        final_frame = max(1.0, float(np.nanmax(frames)))
        ax.set_xlim(0.0, final_frame)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Fraction")
        ax.set_title("Normalized Gaussian maintenance")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(self.root / "gaussian_maintenance_normalized.png", dpi=220)
        plt.close(fig)

    def render(self):
        if not self.enabled:
            return
        rows = self._read_rows(self.root / "loss_history.csv")
        errors = []
        for plotter, args in (
            (self._plot_convergence, (rows,)),
            (self._plot_full_sequence, (rows,)),
            (self._plot_maintenance, ()),
            (self._plot_maintenance_normalized, ()),
        ):
            try:
                plotter(*args)
            except Exception as error:
                errors.append(f"{plotter.__name__}: {error}")
        if errors:
            print("[loss diagnostics] plot warnings: " + "; ".join(errors))

    def finalize(self):
        if not self.enabled:
            return
        self.render()
        print(f"[loss diagnostics] saved to {self.root}")
