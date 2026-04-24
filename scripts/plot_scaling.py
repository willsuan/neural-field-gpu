#!/usr/bin/env python3
"""
Generate scaling plots for the writeup from benchmark CSV output.

Inputs
------
  bench_size.csv  -- header: Nx,Ny,nsteps,total_s,per_step_ms,throughput_Gcell_s
  bench_gpu.csv   -- header: ngpus,wall_s,per_process_avg_s

Outputs
-------
  docs/figures/throughput_vs_size.pdf
  docs/figures/time_per_step_vs_size.pdf
  docs/figures/multi_gpu_scaling.pdf

The time-per-step plot uses log-log axes with a reference O(N^2 log N)
guide line so the measured points compare to a straight line, as the
style guide requests.
"""
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if k not in ("Nx", "Ny", "nsteps", "ngpus")
                         else int(v) for k, v in r.items()})
    return rows


def plot_size_sweep(rows, fig_tps, fig_step):
    N = np.array([r["Nx"] for r in rows])
    throughput = np.array([r["throughput_Gcell_s"] for r in rows])
    step_ms = np.array([r["per_step_ms"] for r in rows])

    # Throughput vs N (linear y, log-2 x)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(N, throughput, "o-", linewidth=1.8, markersize=7, color="#2b6cb0")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel(r"grid side $N$")
    ax.set_ylabel(r"throughput (Gcell/s)")
    ax.set_title("Single-GPU throughput vs. problem size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_tps)
    plt.close(fig)

    # Time per step vs N: log-log with O(N^2 log N) guide line
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.loglog(N, step_ms, "o", markersize=7, color="#2b6cb0",
              label="measured")
    # reference slope: t = c * N^2 * log2(N); fit c at the largest point
    c = step_ms[-1] / (N[-1] ** 2 * math.log2(N[-1]))
    Nr = np.linspace(N[0], N[-1], 80)
    ax.loglog(Nr, c * Nr ** 2 * np.log2(Nr), "--", linewidth=1.3,
              color="#718096", label=r"$O(N^{2}\,\log N)$ reference")
    ax.set_xlabel(r"grid side $N$")
    ax.set_ylabel(r"time per step (ms)")
    ax.set_title("Cost per time step vs. problem size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_step)
    plt.close(fig)


def plot_multi_gpu(rows, fig_path):
    k = np.array([r["ngpus"] for r in rows])
    wall = np.array([r["wall_s"] for r in rows])
    # ideal: wall_k = wall_1 (independent jobs run concurrently)
    ideal = np.full_like(wall, wall[0])
    # effective speedup of the aggregate workload vs. running serially:
    aggregate_serial = wall[0] * k
    speedup = aggregate_serial / wall

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    a1.plot(k, wall, "o-", label="measured wall time", color="#2b6cb0",
            linewidth=1.8, markersize=7)
    a1.plot(k, ideal, "--", label="ideal (constant)", color="#718096",
            linewidth=1.2)
    a1.set_xlabel("concurrent GPUs")
    a1.set_ylabel("wall time (s)")
    a1.set_title("(a) wall time for $K$ concurrent runs")
    a1.set_xticks(k)
    a1.grid(True, ls=":", alpha=0.5)
    a1.legend(frameon=False, fontsize=9)

    a2.plot(k, speedup, "o-", color="#2b6cb0", linewidth=1.8, markersize=7,
            label="effective speedup")
    a2.plot(k, k, "--", color="#718096", linewidth=1.2, label="ideal linear")
    a2.set_xlabel("concurrent GPUs")
    a2.set_ylabel(r"speedup $= K\,T_1 / T_K$")
    a2.set_title("(b) effective throughput speedup")
    a2.set_xticks(k)
    a2.grid(True, ls=":", alpha=0.5)
    a2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main():
    size_csv = ROOT / "bench_size.csv"
    gpu_csv = ROOT / "bench_gpu.csv"

    if not size_csv.exists() or not gpu_csv.exists():
        print("missing bench_size.csv or bench_gpu.csv; run scripts/benchmark.sh first",
              file=sys.stderr)
        sys.exit(1)

    size_rows = read_csv(size_csv)
    gpu_rows = read_csv(gpu_csv)

    plot_size_sweep(size_rows,
                    FIGDIR / "throughput_vs_size.pdf",
                    FIGDIR / "time_per_step_vs_size.pdf")
    plot_multi_gpu(gpu_rows, FIGDIR / "multi_gpu_scaling.pdf")

    print(f"wrote {FIGDIR / 'throughput_vs_size.pdf'}")
    print(f"wrote {FIGDIR / 'time_per_step_vs_size.pdf'}")
    print(f"wrote {FIGDIR / 'multi_gpu_scaling.pdf'}")


if __name__ == "__main__":
    main()
