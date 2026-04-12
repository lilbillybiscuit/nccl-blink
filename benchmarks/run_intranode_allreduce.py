#!/usr/bin/env python3
"""Run intra-node nccl-tests all-reduce sweeps and export CSV.

This script intentionally measures existing NCCL behavior only.
It does not enable Blink or any other experimental path.

It is intentionally simple:
  - it drives nccl-tests `all_reduce_perf`
  - it sweeps GPU counts from 1..N on a single node
  - it writes one raw CSV row per nccl-tests data line
  - it also writes a small summary CSV with the best observed bandwidth per run

Typical usage on an 8x H200 node:
  python3 benchmarks/run_intranode_allreduce.py

More exhaustive data:
  python3 benchmarks/run_intranode_allreduce.py --subset-mode all-combinations
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "benchmarks" / "nccl-tests" / "build" / "all_reduce_perf"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_LIB_DIR = REPO_ROOT / "build" / "lib"

DATA_RE = re.compile(
    r"^\s*(\d+)\s+"       # size (B)
    r"(\d+)\s+"           # count
    r"(\w+)\s+"           # type
    r"(\w+)\s+"           # redop
    r"(-?\d+)\s+"         # root
    r"([\d.]+)\s+"        # time (us) out-of-place
    r"([\d.]+)\s+"        # algbw (GB/s)
    r"([\d.]+)\s+"        # busbw (GB/s)
    r"(\d+|N/A)\s+"       # errors
    r"([\d.]+)\s+"        # time (us) in-place
    r"([\d.]+)\s+"        # algbw (GB/s) in-place
    r"([\d.]+)\s+"        # busbw (GB/s) in-place
    r"(\d+|N/A)"          # errors in-place
)


@dataclass(frozen=True)
class RunConfig:
    gpu_count: int
    visible_devices: str
    subset_index: int
    subset_mode: str


def parse_gpu_counts(spec: str, detected_gpu_count: int) -> List[int]:
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"invalid gpu-count range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    counts = sorted(v for v in values if 1 <= v <= detected_gpu_count)
    if not counts:
        raise ValueError(
            f"gpu-counts={spec!r} produced no valid counts in [1, {detected_gpu_count}]"
        )
    return counts


def parse_size_token(token: str) -> int:
    token = token.strip().upper()
    units = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda item: -len(item[0])):
        if token.endswith(suffix):
            number = float(token[: -len(suffix)])
            return int(number * scale)
    return int(token)


def detect_gpu_count() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to run nvidia-smi -L: {exc.stderr.strip()}") from exc

    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
    if not lines:
        raise RuntimeError("nvidia-smi -L returned no GPUs")
    return len(lines)


def generate_subsets(total_gpus: int, gpu_count: int, mode: str) -> List[List[int]]:
    if mode == "prefix":
        return [list(range(gpu_count))]
    if mode == "all-combinations":
        return [list(combo) for combo in itertools.combinations(range(total_gpus), gpu_count)]
    raise ValueError(f"unknown subset mode: {mode}")


def ensure_binary_exists(binary: Path) -> None:
    if not binary.is_file():
        raise FileNotFoundError(
            f"nccl-tests binary not found: {binary}\n"
            f"Build it first, for example with benchmarks/setup.sh."
        )


def build_env(visible_devices: str, lib_dir: Path, extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["NCCL_BLINK"] = "0"
    env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
    env["NCCL_IB_DISABLE"] = env.get("NCCL_IB_DISABLE", "1")
    env["NCCL_IGNORE_CPU_AFFINITY"] = env.get("NCCL_IGNORE_CPU_AFFINITY", "1")

    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{ld_library_path}" if ld_library_path else str(lib_dir)
    )

    for key, value in extra_env.items():
        env[key] = value
    return env


def parse_perf_output(output: str) -> List[dict[str, object]]:
    def parse_error_field(value: str) -> int | None:
        return None if value == "N/A" else int(value)

    rows = []
    for line in output.splitlines():
        match = DATA_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "size_bytes": int(match.group(1)),
                "size_mb": int(match.group(1)) / (1024 * 1024),
                "count": int(match.group(2)),
                "dtype": match.group(3),
                "redop": match.group(4),
                "root": int(match.group(5)),
                "time_us": float(match.group(6)),
                "algbw_gbps": float(match.group(7)),
                "busbw_gbps": float(match.group(8)),
                "errors": parse_error_field(match.group(9)),
                "time_us_inplace": float(match.group(10)),
                "algbw_gbps_inplace": float(match.group(11)),
                "busbw_gbps_inplace": float(match.group(12)),
                "errors_inplace": parse_error_field(match.group(13)),
            }
        )
    return rows


def summarize_run(rows: List[dict[str, object]], run: RunConfig, log_file: Path) -> dict[str, object]:
    best_oop = max(rows, key=lambda row: row["busbw_gbps"])
    best_inplace = max(rows, key=lambda row: row["busbw_gbps_inplace"])
    return {
        "gpu_count": run.gpu_count,
        "visible_devices": run.visible_devices,
        "subset_index": run.subset_index,
        "subset_mode": run.subset_mode,
        "best_busbw_gbps": best_oop["busbw_gbps"],
        "best_busbw_size_bytes": best_oop["size_bytes"],
        "best_algbw_gbps": best_oop["algbw_gbps"],
        "best_busbw_gbps_inplace": best_inplace["busbw_gbps_inplace"],
        "best_busbw_size_bytes_inplace": best_inplace["size_bytes"],
        "best_algbw_gbps_inplace": best_inplace["algbw_gbps_inplace"],
        "log_file": str(log_file),
    }


def append_rows(csv_path: Path, rows: List[dict[str, object]]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    binary: Path,
    results_dir: Path,
    lib_dir: Path,
    begin_size: str,
    end_size: str,
    factor: str,
    warmup_iters: int,
    iters: int,
    validation: int,
    run: RunConfig,
    extra_env: dict[str, str],
) -> tuple[List[dict[str, object]], Path]:
    env = build_env(run.visible_devices, lib_dir, extra_env)
    command = [
        str(binary),
        "-b", begin_size,
        "-e", end_size,
        "-f", factor,
        "-g", str(run.gpu_count),
        "-c", str(validation),
        "-w", str(warmup_iters),
        "-n", str(iters),
    ]

    proc = subprocess.run(
        command,
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    log_name = f"intranode_allreduce_g{run.gpu_count}_subset{run.subset_index:03d}.log"
    log_file = results_dir / "logs" / log_name
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))

    if proc.returncode != 0:
        raise RuntimeError(
            f"all_reduce_perf failed for GPUs {run.visible_devices}. "
            f"See {log_file}"
        )

    perf_rows = parse_perf_output(proc.stdout)
    if not perf_rows:
        raise RuntimeError(
            f"no performance rows were parsed for GPUs {run.visible_devices}. "
            f"See {log_file}"
        )

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()
    annotated_rows = []
    for row in perf_rows:
        annotated_rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "host": host,
                "gpu_count": run.gpu_count,
                "visible_devices": run.visible_devices,
                "subset_index": run.subset_index,
                "subset_mode": run.subset_mode,
                "binary": str(binary),
                "log_file": str(log_file),
                "begin_size": begin_size,
                "end_size": end_size,
                "factor": factor,
                "validation": validation,
                "warmup_iters": warmup_iters,
                "iters": iters,
                **row,
            }
        )
    return annotated_rows, log_file


def parse_extra_env(items: Iterable[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run intra-node NCCL all-reduce bandwidth sweeps and export CSV."
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--lib-dir", type=Path, default=DEFAULT_LIB_DIR)
    parser.add_argument("--gpu-counts", default="1-8")
    parser.add_argument(
        "--subset-mode",
        choices=("prefix", "all-combinations"),
        default="prefix",
        help="prefix is fast; all-combinations is exhaustive for a given GPU count.",
    )
    parser.add_argument("--begin-size", default="8")
    parser.add_argument("--end-size", default="8G")
    parser.add_argument("--factor", default="2")
    parser.add_argument("--validation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variables passed to all_reduce_perf",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    ensure_binary_exists(args.binary)

    detected_gpu_count = detect_gpu_count()
    gpu_counts = parse_gpu_counts(args.gpu_counts, detected_gpu_count)
    extra_env = parse_extra_env(args.env)

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_csv = results_dir / f"intranode_allreduce_raw_{timestamp}.csv"
    summary_csv = results_dir / f"intranode_allreduce_summary_{timestamp}.csv"

    summary_rows = []

    print(f"Detected GPUs: {detected_gpu_count}")
    print(f"GPU counts: {gpu_counts}")
    print(f"Subset mode: {args.subset_mode}")
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Validation: {args.validation}")
    print()

    for gpu_count in gpu_counts:
        subsets = generate_subsets(detected_gpu_count, gpu_count, args.subset_mode)
        print(f"gpu_count={gpu_count}: running {len(subsets)} subset(s)")
        for subset_index, subset in enumerate(subsets):
            visible_devices = ",".join(str(gpu) for gpu in subset)
            run = RunConfig(
                gpu_count=gpu_count,
                visible_devices=visible_devices,
                subset_index=subset_index,
                subset_mode=args.subset_mode,
            )
            print(
                f"  subset {subset_index + 1}/{len(subsets)}: "
                f"CUDA_VISIBLE_DEVICES={visible_devices}"
            )
            rows, log_file = run_benchmark(
                binary=args.binary.resolve(),
                results_dir=results_dir,
                lib_dir=args.lib_dir.resolve(),
                begin_size=args.begin_size,
                end_size=args.end_size,
                factor=args.factor,
                validation=args.validation,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                run=run,
                extra_env=extra_env,
            )
            append_rows(raw_csv, rows)
            summary_rows.append(summarize_run(rows, run, log_file))

    if summary_rows:
        append_rows(summary_csv, summary_rows)

    print()
    print("Done.")
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
