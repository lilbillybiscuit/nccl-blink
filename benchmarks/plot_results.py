#!/usr/bin/env python3
"""Generate paper-style comparison plots from parsed benchmark data.

Reads:
  results/performance.csv  — AllReduce throughput data
  results/treegen_timing.csv — Blink TreeGen computation time

Generates:
  results/figures/allreduce_bw_vs_size.pdf  — BW vs message size per topology
  results/figures/allreduce_bw_vs_topo.pdf  — BW vs topology at fixed size
  results/figures/treegen_timing.pdf        — TreeGen phase breakdown

Usage:
  python plot_results.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Style
COLORS = {'NCCL (default)': '#1f77b4', 'NCCL + Blink': '#d62728'}
TOPO_ORDER = ['8gpu', '6gpu', '5gpu', '3gpu']
TOPO_LABELS = {
    '8gpu': '8 GPU\n(full)',
    '6gpu': '6 GPU\n(frag)',
    '5gpu': '5 GPU\n(frag)',
    '3gpu': '3 GPU\n(frag)',
    'local_2gpu': '2 GPU\n(local)',
}


def plot_bw_vs_size(df):
    """AllReduce bus bandwidth vs message size, one subplot per topology."""
    topos = [t for t in TOPO_ORDER if t in df['topology'].unique()]
    if not topos:
        topos = sorted(df['topology'].unique())

    n = len(topos)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, topo in zip(axes, topos):
        for blink_val, label in [(0, 'NCCL (default)'), (1, 'NCCL + Blink')]:
            subset = df[(df['topology'] == topo) & (df['blink'] == blink_val)]
            if subset.empty:
                continue
            ax.plot(subset['size_mb'], subset['busbw_gbps'],
                    'o-', label=label, color=COLORS[label], markersize=4)

        ax.set_xlabel('Message Size (MB)')
        ax.set_xscale('log', base=2)
        ax.set_title(TOPO_LABELS.get(topo, topo))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Bus Bandwidth (GB/s)')
    fig.suptitle('AllReduce Throughput: NCCL vs Blink', fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / 'allreduce_bw_vs_size.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_bw_vs_topo(df, target_size_mb=128):
    """Bar chart: AllReduce bandwidth at a fixed message size, grouped by topology."""
    # Find closest available size
    sizes = sorted(df['size_mb'].unique())
    closest = min(sizes, key=lambda s: abs(s - target_size_mb))

    subset = df[df['size_mb'] == closest]
    topos = [t for t in TOPO_ORDER if t in subset['topology'].unique()]
    if not topos:
        topos = sorted(subset['topology'].unique())

    fig, ax = plt.subplots(figsize=(max(4, len(topos) * 1.5), 4))

    x = range(len(topos))
    width = 0.35

    for i, (blink_val, label) in enumerate([(0, 'NCCL (default)'), (1, 'NCCL + Blink')]):
        vals = []
        for topo in topos:
            row = subset[(subset['topology'] == topo) & (subset['blink'] == blink_val)]
            vals.append(row['busbw_gbps'].mean() if not row.empty else 0)
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=label, color=COLORS[label])

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS.get(t, t) for t in topos])
    ax.set_ylabel('Bus Bandwidth (GB/s)')
    ax.set_title(f'AllReduce @ {closest:.0f}MB: NCCL vs Blink')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / 'allreduce_bw_vs_topo.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_treegen_timing(df):
    """Stacked bar chart of TreeGen phase times."""
    phases = ['extract_us', 'mwu_us', 'refine_us', 'chain_us']
    phase_labels = ['Extract', 'MWU', 'Refine', 'Chain']
    phase_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    # Only Blink=1 has timing data
    df = df[df['blink'] == 1].copy()
    if df.empty:
        print("  No TreeGen timing data (Blink not enabled?)")
        return

    topos = [t for t in TOPO_ORDER if t in df['topology'].unique()]
    if not topos:
        topos = sorted(df['topology'].unique())

    fig, ax = plt.subplots(figsize=(max(4, len(topos) * 1.5), 4))

    x = range(len(topos))
    bottoms = [0.0] * len(topos)

    for phase, label, color in zip(phases, phase_labels, phase_colors):
        vals = []
        for topo in topos:
            rows = df[df['topology'] == topo]
            vals.append(rows[phase].mean() if not rows.empty else 0)
        ax.bar(x, vals, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Add total time labels on top
    for xi, total in zip(x, bottoms):
        ax.text(xi, total + max(bottoms) * 0.02, f'{total:.0f}μs',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS.get(t, t) for t in topos])
    ax.set_ylabel('Time (μs)')
    ax.set_title('Blink TreeGen Computation Time')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / 'treegen_timing.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    perf_csv = RESULTS_DIR / "performance.csv"
    timing_csv = RESULTS_DIR / "treegen_timing.csv"

    if not perf_csv.exists() and not timing_csv.exists():
        print("No CSV data found. Run parse_results.py first.")
        sys.exit(1)

    print("Generating plots...")

    if perf_csv.exists():
        df = pd.read_csv(perf_csv)
        print(f"  Performance data: {len(df)} rows")
        plot_bw_vs_size(df)
        plot_bw_vs_topo(df)

    if timing_csv.exists():
        df_timing = pd.read_csv(timing_csv)
        print(f"  Timing data: {len(df_timing)} rows")
        plot_treegen_timing(df_timing)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
