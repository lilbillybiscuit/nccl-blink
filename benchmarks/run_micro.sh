#!/bin/bash
# Blink Micro-Benchmark Sweep
# Measures AllReduce throughput (GB/s) and latency (us) for:
#   - NCCL_BLINK=0 (default trees) vs NCCL_BLINK=1 (packed spanning trees)
#   - Full and fragmented GPU allocations (via CUDA_VISIBLE_DEVICES)
#   - Message sizes from 1MB to 1000MB (doubling)
#
# Usage:
#   ./run_micro.sh                    # Full sweep: all GPUs + fragmented subsets
#   ./run_micro.sh --quick            # Quick sanity check (1MB only, all GPUs)
#   ./run_micro.sh --gpus 0,1,4      # Test specific GPU subset
#   ./run_micro.sh --full-only       # Only run with all GPUs (no fragmented)
#   ./run_micro.sh --simulated        # Use DGX-1 XML topologies (for testing without NVLink)
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
USE_SIMULATED=0
USE_DEGRADED=0
FULL_ONLY=0
CUSTOM_GPUS=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick) END_SIZE="1M"; ITERS="5"; WARMUP_ITERS="2"; shift;;
    --simulated) USE_SIMULATED=1; shift;;
    --degraded) USE_DEGRADED=1; shift;;
    --full-only) FULL_ONLY=1; shift;;
    --gpus) CUSTOM_GPUS="$2"; shift 2;;
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
echo "Results: $RESULTS_DIR/"
echo ""

run_bench() {
  local TOPO_NAME=$1
  local BLINK=$2
  local NGPUS=$3
  local TOPO_FILE=$4  # empty string = no NCCL_TOPO_FILE
  local CVD=$5        # CUDA_VISIBLE_DEVICES value, empty = all
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
    ENV_VARS+=("NCCL_IGNORE_DISABLED_P2P=2")
    ENV_VARS+=("NCCL_IGNORE_CPU_AFFINITY=1")
  fi

  if [ -n "$CVD" ]; then
    ENV_VARS+=("CUDA_VISIBLE_DEVICES=$CVD")
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

if [ "$USE_DEGRADED" -eq 1 ]; then
  # Degraded H200 topologies: artificially constrained NVLink counts
  echo "Degraded mode: using modified H200 topology XMLs"
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  for VARIANT in twoisland sparse; do
    TOPO_FILE="$TOPO_DIR/h200_${VARIANT}.xml"
    if [ ! -f "$TOPO_FILE" ]; then
      echo "Warning: $TOPO_FILE not found (skipping)"
      continue
    fi
    echo "=== Degraded topology: $VARIANT ==="
    for BLINK in 0 1; do
      run_bench "degraded_${VARIANT}" "$BLINK" "$NGPUS" "$TOPO_FILE" "" 2>&1 | tee -a "$SUMMARY"
    done
  done
elif [ "$USE_SIMULATED" -eq 1 ]; then
  # Simulated topologies via NCCL_TOPO_FILE (for machines without NVLink)
  echo "Simulated mode: using DGX-1 topology XML files"
  for TOPO in 8gpu 6gpu 5gpu 3gpu; do
    TOPO_FILE="$TOPO_DIR/dgx1_${TOPO}.xml"
    if [ ! -f "$TOPO_FILE" ]; then
      echo "Warning: topology file not found: $TOPO_FILE (skipping)"
      continue
    fi
    NGPUS=$(echo "$TOPO" | grep -o '[0-9]*')
    for BLINK in 0 1; do
      run_bench "sim_${TOPO}" "$BLINK" "$NGPUS" "$TOPO_FILE" "" 2>&1 | tee -a "$SUMMARY"
    done
  done
elif [ -n "$CUSTOM_GPUS" ]; then
  # Custom GPU subset
  NGPUS=$(echo "$CUSTOM_GPUS" | tr ',' '\n' | wc -l)
  echo "Custom GPU subset: $CUSTOM_GPUS ($NGPUS GPUs)"
  for BLINK in 0 1; do
    run_bench "custom_${NGPUS}gpu" "$BLINK" "$NGPUS" "" "$CUSTOM_GPUS"
  done
else
  # Real hardware mode (default)
  TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  echo "Real hardware mode: $TOTAL_GPUS GPUs detected"
  echo ""

  # Full allocation
  echo "=== Full allocation: all $TOTAL_GPUS GPUs ==="
  for BLINK in 0 1; do
    run_bench "full_${TOTAL_GPUS}gpu" "$BLINK" "$TOTAL_GPUS" "" ""
  done

  if [ "$FULL_ONLY" -eq 0 ] && [ "$TOTAL_GPUS" -ge 4 ]; then
    # Fragmented allocations via CUDA_VISIBLE_DEVICES
    # These simulate scheduler fragmentation on real NVLink hardware

    # ~75% allocation (e.g., 6 of 8)
    FRAG6=$(seq 0 $(($TOTAL_GPUS - 3)) | tr '\n' ',')
    FRAG6=${FRAG6%,}
    NGPUS6=$(echo "$FRAG6" | tr ',' '\n' | wc -l)
    echo "=== Fragmented: $NGPUS6 GPUs ($FRAG6) ==="
    for BLINK in 0 1; do
      run_bench "frag_${NGPUS6}gpu" "$BLINK" "$NGPUS6" "" "$FRAG6"
    done

    # ~62% allocation (e.g., 5 of 8)
    FRAG5=$(seq 0 $(($TOTAL_GPUS - 4)) | tr '\n' ',')
    FRAG5=${FRAG5%,}
    NGPUS5=$(echo "$FRAG5" | tr ',' '\n' | wc -l)
    echo "=== Fragmented: $NGPUS5 GPUs ($FRAG5) ==="
    for BLINK in 0 1; do
      run_bench "frag_${NGPUS5}gpu" "$BLINK" "$NGPUS5" "" "$FRAG5"
    done

    # Adversarial: cross-NUMA GPUs (first, second, and one from other NUMA)
    FRAG3="0,1,$((TOTAL_GPUS - 1))"
    echo "=== Adversarial: 3 GPUs ($FRAG3) ==="
    for BLINK in 0 1; do
      run_bench "frag_3gpu" "$BLINK" "3" "" "$FRAG3"
    done
  fi
fi

echo ""
echo "=== Sweep complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Summary: $SUMMARY"
echo ""
echo "Parse results with:  python parse_results.py"
echo "Generate plots with: python plot_results.py"
