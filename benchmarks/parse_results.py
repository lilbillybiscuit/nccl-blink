#!/usr/bin/env python3
"""Parse nccl-tests output logs into structured CSV files.

nccl-tests output format (all_reduce_perf):
  #                                                              out-of-place                       in-place
  #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
  #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)

This also extracts Blink timing lines from NCCL_DEBUG=GRAPH output:
  Blink timing: extract=X.Xus MWU=X.Xus refine=X.Xus chain=X.Xus total=X.Xus

Usage:
  python parse_results.py                     # Parse all logs in results/
  python parse_results.py results/micro_*.log # Parse specific files
"""
import sys
import os
import re
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Regex for nccl-tests data lines
# Example: "     1048576      262144     float     sum      -1    123.4    8.51   14.89      0    122.1    8.59   15.03      0"
DATA_RE = re.compile(
    r'^\s*(\d+)\s+'       # size (B)
    r'(\d+)\s+'           # count (elements)
    r'(\w+)\s+'           # type
    r'(\w+)\s+'           # redop
    r'(-?\d+)\s+'         # root
    r'([\d.]+)\s+'        # time (us) out-of-place
    r'([\d.]+)\s+'        # algbw (GB/s)
    r'([\d.]+)\s+'        # busbw (GB/s)
    r'(\d+)\s+'           # wrong
    r'([\d.]+)\s+'        # time (us) in-place
    r'([\d.]+)\s+'        # algbw (GB/s) in-place
    r'([\d.]+)\s+'        # busbw (GB/s) in-place
    r'(\d+)'              # wrong in-place
)

# Regex for Blink timing lines
TIMING_RE = re.compile(
    r'Blink timing: extract=([\d.]+)us MWU=([\d.]+)us refine=([\d.]+)us chain=([\d.]+)us total=([\d.]+)us'
)


def parse_filename(path):
    """Extract topology and blink setting from filename.
    Expected: micro_<topo>_blink<0|1>.log
    """
    name = Path(path).stem
    m = re.match(r'micro_(\w+)_blink(\d)', name)
    if m:
        return m.group(1), int(m.group(2))
    return name, -1


def parse_log(filepath):
    """Parse a single nccl-tests log file.
    Returns (data_rows, timing_rows).
    """
    topo, blink = parse_filename(filepath)
    data_rows = []
    timing_rows = []

    with open(filepath) as f:
        for line in f:
            # Check for performance data
            m = DATA_RE.match(line)
            if m:
                data_rows.append({
                    'topology': topo,
                    'blink': blink,
                    'size_bytes': int(m.group(1)),
                    'size_mb': int(m.group(1)) / (1024 * 1024),
                    'count': int(m.group(2)),
                    'type': m.group(3),
                    'redop': m.group(4),
                    'time_us': float(m.group(6)),
                    'algbw_gbps': float(m.group(7)),
                    'busbw_gbps': float(m.group(8)),
                    'errors': int(m.group(9)),
                    'time_us_inplace': float(m.group(10)),
                    'algbw_gbps_inplace': float(m.group(11)),
                    'busbw_gbps_inplace': float(m.group(12)),
                    'errors_inplace': int(m.group(13)),
                })
                continue

            # Check for Blink timing
            m = TIMING_RE.search(line)
            if m:
                timing_rows.append({
                    'topology': topo,
                    'blink': blink,
                    'extract_us': float(m.group(1)),
                    'mwu_us': float(m.group(2)),
                    'refine_us': float(m.group(3)),
                    'chain_us': float(m.group(4)),
                    'total_us': float(m.group(5)),
                })

    return data_rows, timing_rows


def main():
    if len(sys.argv) > 1:
        log_files = sys.argv[1:]
    else:
        log_files = sorted(RESULTS_DIR.glob("micro_*.log"))
        if not log_files:
            print(f"No log files found in {RESULTS_DIR}/")
            print("Run benchmarks first: ./run_micro.sh")
            sys.exit(1)

    all_data = []
    all_timing = []

    for f in log_files:
        print(f"Parsing {f}...")
        data, timing = parse_log(f)
        all_data.extend(data)
        all_timing.extend(timing)
        print(f"  {len(data)} data rows, {len(timing)} timing entries")

    # Write performance CSV
    if all_data:
        perf_csv = RESULTS_DIR / "performance.csv"
        with open(perf_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_data[0].keys())
            w.writeheader()
            w.writerows(all_data)
        print(f"\nPerformance data: {perf_csv} ({len(all_data)} rows)")

    # Write timing CSV
    if all_timing:
        timing_csv = RESULTS_DIR / "treegen_timing.csv"
        with open(timing_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_timing[0].keys())
            w.writeheader()
            w.writerows(all_timing)
        print(f"TreeGen timing:   {timing_csv} ({len(all_timing)} rows)")

    if not all_data and not all_timing:
        print("No data parsed from any log file.")
        sys.exit(1)


if __name__ == '__main__':
    main()
