#!/bin/bash
################################################################################
# Setup Script for TensorRT Profiling
#
# Sets up the profiling environment on a local Blackwell workstation.
# Uses Docker containers for all tools (TensorRT, Nsight Systems, Nsight Compute)
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit
#   - NVIDIA GPU with driver installed
#
# Usage:
#   ./scripts/setup.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Container image (contains TensorRT, Nsight tools, and ModelOpt)
CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"

echo "================================================================================"
echo "TENSORRT PROFILING SETUP"
echo "================================================================================"
echo ""

#-------------------------------------------------------------------------------
# 1. System Check
#-------------------------------------------------------------------------------
echo ">>> Step 1: System Check"
echo ""

# Check GPU
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    
    echo "GPU:              $GPU_NAME (x$GPU_COUNT)"
    echo "Driver:           $DRIVER_VERSION"
    echo "CUDA (Driver):    $CUDA_VERSION"
    
    # Check for Blackwell
    if [[ "$GPU_NAME" == *"Blackwell"* ]] || [[ "$GPU_NAME" == *"RTX PRO 6000"* ]] || [[ "$GPU_NAME" == *"B100"* ]] || [[ "$GPU_NAME" == *"B200"* ]]; then
        echo "Architecture:     Blackwell ✓"
        echo ""
        echo "  Expected Blackwell optimizations:"
        echo "    - Native MXFP8 (E4M3/E5M2) support"
        echo "    - Native NVFP4 support"
        echo "    - 4th generation Tensor Cores"
    else
        echo "Architecture:     $(echo $GPU_NAME | grep -oP 'RTX \d+|A\d+|H\d+|V\d+' | head -1 || echo 'Unknown')"
    fi
else
    echo "ERROR: nvidia-smi not found. Ensure GPU driver is installed."
    exit 1
fi

echo ""

#-------------------------------------------------------------------------------
# 2. Docker Check
#-------------------------------------------------------------------------------
echo ">>> Step 2: Docker Check"
echo ""

if docker info &>/dev/null; then
    echo "Docker:           $(docker --version | cut -d' ' -f3)"
    
    # Check NVIDIA Container Toolkit
    if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &>/dev/null; then
        echo "NVIDIA Runtime:   Available ✓"
    else
        echo "ERROR: NVIDIA Container Toolkit not configured properly."
        echo "Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
else
    echo "ERROR: Docker not available."
    echo "Install Docker and NVIDIA Container Toolkit."
    exit 1
fi

echo ""

#-------------------------------------------------------------------------------
# 3. Pull Container Image
#-------------------------------------------------------------------------------
echo ">>> Step 3: Container Image"
echo ""

echo "Container:        $CONTAINER_IMAGE"

if docker image inspect "$CONTAINER_IMAGE" &>/dev/null; then
    echo "Status:           Already pulled ✓"
else
    echo "Status:           Pulling..."
    docker pull "$CONTAINER_IMAGE"
    echo "Status:           Pulled ✓"
fi

# Verify container contents
echo ""
echo "Verifying container tools..."
docker run --rm "$CONTAINER_IMAGE" bash -c "
    echo '  TensorRT:       '$(pip show tensorrt 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'N/A')
    echo '  ModelOpt:       '$(pip show nvidia-modelopt 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'N/A')
    echo '  nsys:           '$(which nsys 2>/dev/null && echo '✓' || echo '✗')
    echo '  ncu:            '$(which ncu 2>/dev/null && echo '✓' || echo '✗')
    echo '  trtexec:        '$(which trtexec 2>/dev/null && echo '✓' || echo '✗')
" 2>/dev/null || echo "  (Verification skipped)"

echo ""

#-------------------------------------------------------------------------------
# 4. Create Directory Structure
#-------------------------------------------------------------------------------
echo ">>> Step 4: Directory Structure"
echo ""

mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/src"
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/tools"
mkdir -p "$PROJECT_ROOT/results/nsight-systems"
mkdir -p "$PROJECT_ROOT/results/nsight-compute"
mkdir -p "$PROJECT_ROOT/results/benchmark"
mkdir -p "$PROJECT_ROOT/engines"
mkdir -p "$PROJECT_ROOT/logs"

echo "Directory structure created ✓"
echo ""

#-------------------------------------------------------------------------------
# 5. Verify Models
#-------------------------------------------------------------------------------
echo ">>> Step 5: Verify Models"
echo ""

for model in vit_fp16_bs_064.onnx vit_mxfp8_bs_064.onnx vit_nvfp4_bs_064.onnx; do
    if [[ -f "$PROJECT_ROOT/models/$model" ]]; then
        SIZE=$(du -h "$PROJECT_ROOT/models/$model" | cut -f1)
        echo "  ✓ $model ($SIZE)"
    else
        echo "  ✗ $model NOT FOUND"
    fi
done

echo ""

#-------------------------------------------------------------------------------
# 6. Make Scripts Executable
#-------------------------------------------------------------------------------
echo ">>> Step 6: Script Permissions"
echo ""

chmod +x "$PROJECT_ROOT/scripts/"*.sh 2>/dev/null || true
chmod +x "$PROJECT_ROOT/tools/"*.sh 2>/dev/null || true

echo "Scripts made executable ✓"
echo ""

#-------------------------------------------------------------------------------
# 7. Summary
#-------------------------------------------------------------------------------
echo "================================================================================"
echo "SETUP COMPLETE"
echo "================================================================================"
echo ""
echo "GPU:              $GPU_NAME (x$GPU_COUNT)"
echo "Container:        $CONTAINER_IMAGE"
echo "Project Root:     $PROJECT_ROOT"
echo ""
echo "Quick Start:"
echo ""
echo "  # Run full profiling (nsys + ncu + benchmark)"
echo "  ./scripts/run_profiling.sh"
echo ""
echo "  # Run only Nsight Systems profiling"
echo "  ./scripts/run_profiling.sh --nsys"
echo ""
echo "  # Run only Nsight Compute profiling"
echo "  ./scripts/run_profiling.sh --ncu"
echo ""
echo "  # Run benchmark only (fastest)"
echo "  ./scripts/run_profiling.sh --benchmark"
echo ""
echo "Results will be saved to:"
echo "  - Nsight Systems: $PROJECT_ROOT/results/nsight-systems/"
echo "  - Nsight Compute: $PROJECT_ROOT/results/nsight-compute/"
echo "  - Benchmarks:     $PROJECT_ROOT/results/benchmark/"
echo "  - TRT Engines:    $PROJECT_ROOT/engines/"
echo ""
echo "================================================================================"
