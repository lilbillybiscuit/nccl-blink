#!/bin/bash
# Setup script for Blink micro-benchmarks
# Clones nccl-tests and builds against our modified NCCL
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCCL_HOME="$(cd "$SCRIPT_DIR/.." && pwd)/build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== Blink Benchmark Setup ==="
echo "NCCL_HOME: $NCCL_HOME"
echo "CUDA_HOME: $CUDA_HOME"

# 1. Build NCCL if not already built
if [ ! -f "$NCCL_HOME/lib/libnccl.so" ]; then
  echo ""
  echo "--- Building NCCL ---"
  cd "$SCRIPT_DIR/.."
  make -j$(nproc) src.build CUDA_HOME="$CUDA_HOME"
fi

echo ""
echo "--- NCCL library: $(ls -la $NCCL_HOME/lib/libnccl.so*) ---"

# 2. Clone and build nccl-tests
NCCL_TESTS_DIR="$SCRIPT_DIR/nccl-tests"
if [ ! -d "$NCCL_TESTS_DIR" ]; then
  echo ""
  echo "--- Cloning nccl-tests ---"
  git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_DIR"
fi

echo ""
echo "--- Building nccl-tests ---"
cd "$NCCL_TESTS_DIR"
make MPI=0 NCCL_HOME="$NCCL_HOME" CUDA_HOME="$CUDA_HOME" -j$(nproc)

echo ""
echo "--- Verifying build ---"
ls -la "$NCCL_TESTS_DIR/build/"*_perf

# 3. Create results directory
mkdir -p "$SCRIPT_DIR/results/figures"

# 4. Install Python deps (for plotting)
echo ""
echo "--- Installing Python dependencies ---"
pip install --quiet matplotlib pandas 2>/dev/null || {
  echo "Warning: pip install failed. Plot generation requires matplotlib and pandas."
  echo "Install manually: pip install matplotlib pandas"
}

echo ""
echo "=== Setup complete ==="
echo "Run benchmarks with: cd $SCRIPT_DIR && bash run_micro.sh"
