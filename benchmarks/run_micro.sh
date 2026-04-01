#!/bin/bash
# Blink Micro-Benchmark Sweep
# Measures AllReduce throughput (GB/s) and latency (us) for:
#   - NCCL_BLINK=0 (default trees) vs NCCL_BLINK=1 (packed spanning trees)
#   - Multiple topology configurations (full 8-GPU, fragmented 6/5/3)
#   - Message sizes from 1MB to 1000MB (doubling)
#
# Usage:
#   ./run_micro.sh                    # Run full sweep
#   ./run_micro.sh --topo 8gpu        # Run only 8-GPU topology
#   ./run_micro.sh --quick            # Quick sanity check (1MB only)
#   ./run_micro.sh --local            # Use local GPUs (no NCCL_TOPO_FILE)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCCL_HOME="$(cd "$SCRIPT_DIR/.." && pwd)/build"
NCCL_TESTS="$SCRIPT_DIR/nccl-tests"
RESULTS_DIR="$SCRIPT_DIR/results"
TOPO_DIR="$SCRIPT_DIR/topologies"

# Defaults
BEGIN_SIZE="1M"
END_SIZE="1000M"
FACTOR="2"
WARMUP_ITERS="5"
ITERS="20"
TOPOS="8gpu 6gpu 5gpu 3gpu"
USE_LOCAL=0

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --topo) TOPOS="$2"; shift 2;;
    --quick) END_SIZE="1M"; ITERS="5"; WARMUP_ITERS="2"; shift;;
    --local) USE_LOCAL=1; shift;;
    --iters) ITERS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Verify nccl-tests is built
if [ ! -f "$NCCL_TESTS/build/all_reduce_perf" ]; then
  echo "Error: nccl-tests not built. Run ./setup.sh first."
  exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Blink Micro-Benchmark Sweep ==="
echo "Sizes: $BEGIN_SIZE to $END_SIZE (x$FACTOR)"
echo "Iters: $ITERS (warmup: $WARMUP_ITERS)"
echo "Topologies: $TOPOS"
echo "Results: $RESULTS_DIR/"
echo ""

run_bench() {
  local TOPO_NAME=$1
  local BLINK=$2
  local NGPUS=$3
  local TOPO_FILE=$4
  local OUTFILE="$RESULTS_DIR/micro_${TOPO_NAME}_blink${BLINK}.log"

  echo "--- Running: ${TOPO_NAME} BLINK=${BLINK} (${NGPUS} GPUs) ---"

  local ENV_VARS=(
    "LD_LIBRARY_PATH=$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
    "NCCL_BLINK=$BLINK"
    "NCCL_DEBUG=GRAPH"
    "NCCL_DEBUG_SUBSYS=GRAPH"
  )

  if [ -n "$TOPO_FILE" ]; then
    ENV_VARS+=("NCCL_TOPO_FILE=$TOPO_FILE")
  fi

  env "${ENV_VARS[@]}" \
    "$NCCL_TESTS/build/all_reduce_perf" \
      -b "$BEGIN_SIZE" -e "$END_SIZE" -f "$FACTOR" \
      -g "$NGPUS" \
      -n "$ITERS" -w "$WARMUP_ITERS" \
      2>&1 | tee "$OUTFILE"

  echo ""
  echo "  -> Saved to $OUTFILE"
  echo ""
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$RESULTS_DIR/summary_${TIMESTAMP}.txt"
echo "Blink Micro-Benchmark Run: $(date)" > "$SUMMARY"
echo "Host: $(hostname)" >> "$SUMMARY"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'unknown')" >> "$SUMMARY"
echo "" >> "$SUMMARY"

if [ "$USE_LOCAL" -eq 1 ]; then
  # Use actual local GPUs (no topology simulation)
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  echo "Local mode: $NGPUS GPUs detected"
  for BLINK in 0 1; do
    run_bench "local_${NGPUS}gpu" "$BLINK" "$NGPUS" ""
  done
else
  # Simulated topologies via NCCL_TOPO_FILE
  for TOPO in $TOPOS; do
    TOPO_FILE="$TOPO_DIR/dgx1_${TOPO}.xml"
    if [ ! -f "$TOPO_FILE" ]; then
      echo "Warning: topology file not found: $TOPO_FILE (skipping)"
      continue
    fi
    NGPUS=$(echo "$TOPO" | grep -o '[0-9]*')

    for BLINK in 0 1; do
      run_bench "$TOPO" "$BLINK" "$NGPUS" "$TOPO_FILE" 2>&1 | tee -a "$SUMMARY"
    done
  done
fi

echo ""
echo "=== Sweep complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Summary: $SUMMARY"
echo ""
echo "Parse results with:  python parse_results.py"
echo "Generate plots with: python plot_results.py"
