# ONNX Model Profiling with TensorRT and Nsight Tools

This project profiles the performance of Vision Transformer (ViT) models across different precision formats (FP16, MXFP8, NVFP4) using NVIDIA's TensorRT optimization and Nsight profiling tools.

## Table of Contents
- [Quick Start Guide](#quick-start-guide)
- [Overview](#overview)
- [TensorRT Deep Dive](#tensorrt-deep-dive)
- [What is a TensorRT Engine?](#what-is-a-tensorrt-engine)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Profiling Your Own Models](#profiling-your-own-models)
- [Results](#results)

---

## Quick Start Guide

**Complete step-by-step walkthrough for replicating this profiling setup on a Blackwell GPU.**

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **GPU** | NVIDIA Blackwell (RTX 5000 series, RTX PRO 6000) | RTX PRO 6000 |
| **Driver** | 550+ | 580+ |
| **OS** | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 |
| **Docker** | 24.0+ | Latest |
| **Disk Space** | 50 GB | 100 GB |
| **RAM** | 32 GB | 64 GB |

### Step 1: Verify GPU and Driver

```bash
# Check GPU is detected
nvidia-smi

# Expected output should show Blackwell GPU:
# NVIDIA RTX PRO 6000 Blackwell Workstation Edition
# Driver Version: 580.xx.xx   CUDA Version: 13.x

# Verify compute capability (Blackwell = SM 12.0)
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Step 2: Install Docker with NVIDIA Container Toolkit

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

### Step 3: Clone This Repository

```bash
cd ~/repos  # or your preferred directory
git clone <repository-url> profiling_blackwell
cd profiling_blackwell
```

### Step 4: Pull the Container Image

```bash
# Pull the NVIDIA PyTorch container (includes TensorRT, Nsight tools)
# This is ~15 GB, may take 10-20 minutes
docker pull nvcr.io/nvidia/pytorch:25.06-py3

# Verify the container works
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "python3 -c 'import tensorrt; print(f\"TensorRT: {tensorrt.__version__}\")'"
```

### Step 5: Prepare Your ONNX Models

Place your ONNX models in the `models/` directory:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Copy your models (example)
cp /path/to/your/model_fp16.onnx models/
cp /path/to/your/model_nvfp4.onnx models/

# Verify models are in place
ls -lh models/
```

**Model naming convention** (update `scripts/run_profiling.sh` if different):
```
models/
├── vit_fp16_bs_064.onnx    # FP16 baseline
├── vit_mxfp8_bs_064.onnx   # MXFP8 (optional - requires plugin)
└── vit_nvfp4_bs_064.onnx   # NVFP4 quantized
```

### Step 6: Configure the Profiling Script

Edit `scripts/run_profiling.sh` to match your models:

```bash
# Open the script
nano scripts/run_profiling.sh

# Find and update the MODELS array (around line 50):
declare -a MODELS=(
    "your_model_fp16.onnx:fp16"
    "your_model_nvfp4.onnx:nvfp4"
)

# Optionally adjust parameters:
WARMUP=50           # Warmup iterations
ITERATIONS=100      # Benchmark iterations
CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
```

### Step 7: Run the Profiling

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Option A: Quick benchmark only (fastest, ~5 min)
./scripts/run_profiling.sh --benchmark

# Option B: Full profiling with Nsight Systems (~15 min)
./scripts/run_profiling.sh --nsys

# Option C: Complete profiling (nsys + ncu + benchmark, ~30 min)
./scripts/run_profiling.sh
```

**Expected output:**
```
[2025-12-04 10:30:00] ================================================================================
[2025-12-04 10:30:00] TENSORRT PROFILING - FP16 / NVFP4
[2025-12-04 10:30:00] ================================================================================
[2025-12-04 10:30:00] >>> Checking environment...
[2025-12-04 10:30:00] GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
[2025-12-04 10:30:00] Docker: 29.x.x
[2025-12-04 10:30:00] Container: nvcr.io/nvidia/pytorch:25.06-py3
...
[2025-12-04 10:35:00] >>> Benchmarking: fp16
[2025-12-04 10:35:05]     Mean Latency:   10.4 ms
[2025-12-04 10:35:05]     Throughput:     95.8 qps
...
```

### Step 8: View Results

```bash
# View the summary report
cat results/runs/*/REPORT.txt

# List all generated files
find results/runs -type f

# Results structure:
results/runs/YYYYMMDD_HHMMSS/
├── REPORT.txt              # Summary comparison
├── benchmark/
│   ├── fp16.json           # FP16 metrics
│   └── nvfp4.json          # NVFP4 metrics
├── nsight-systems/
│   ├── fp16/
│   │   └── profile.nsys-rep  # Open in Nsight Systems GUI
│   └── nvfp4/
│       └── profile.nsys-rep
└── nsight-compute/
    └── ...
```

### Step 9: Analyze Kernel-Level Performance

```bash
# Generate kernel summary from Nsight Systems profile
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  nsys stats --force-export=true \
  /workspace/results/runs/*/nsight-systems/fp16/profile.nsys-rep

# Compare FP16 vs NVFP4 kernel breakdown
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "
    echo '=== FP16 Top Kernels ===' && \
    nsys stats --force-export=true /workspace/results/runs/*/nsight-systems/fp16/profile.nsys-rep 2>&1 | \
    grep -A 20 'CUDA GPU Kernel Summary' && \
    echo '' && \
    echo '=== NVFP4 Top Kernels ===' && \
    nsys stats --force-export=true /workspace/results/runs/*/nsight-systems/nvfp4/profile.nsys-rep 2>&1 | \
    grep -A 20 'CUDA GPU Kernel Summary'
  "
```

### Step 10: View in Nsight Systems GUI (Optional)

To visualize the timeline on a machine with a display:

```bash
# Option A: Copy .nsys-rep files to a machine with Nsight Systems GUI
scp results/runs/*/nsight-systems/*/*.nsys-rep user@workstation:/path/to/view/

# Option B: Install Nsight Systems locally
# Download from: https://developer.nvidia.com/nsight-systems
# Then open: nsys-ui results/runs/*/nsight-systems/fp16/profile.nsys-rep
```

---

### Troubleshooting

<details>
<summary><b>Docker: "permission denied" error</b></summary>

```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```
</details>

<details>
<summary><b>Container: "NVIDIA driver not detected"</b></summary>

```bash
# Reinstall NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```
</details>

<details>
<summary><b>Engine build fails for MXFP8</b></summary>

MXFP8 models require the `TRT_MXFP8DequantizeLinear` plugin which is not included in standard containers. Either:
1. Skip MXFP8 (comment out in `run_profiling.sh`)
2. Use a container with ModelOpt TRT plugins built-in
</details>

<details>
<summary><b>Nsight Compute requires --privileged</b></summary>

The profiling script already includes `--privileged` for NCU. If you still get permission errors:

```bash
# Run container with extended privileges
docker run --rm --gpus all --privileged --cap-add=SYS_ADMIN ...
```
</details>

<details>
<summary><b>Out of GPU memory</b></summary>

Reduce batch size in your ONNX model or adjust the profiling parameters:

```bash
# In run_profiling.sh, reduce iterations
WARMUP=10
ITERATIONS=50
```
</details>

---

## Overview

### Goal
Compare inference performance **before and after TensorRT optimization** across different precision formats:

| Model | Precision | Size | Description |
|-------|-----------|------|-------------|
| `vit_fp16_bs_064.onnx` | FP16 | 166 MB | Half-precision floating point |
| `vit_mxfp8_bs_064.onnx` | MXFP8 | 87 MB | Microscaling FP8 (NVIDIA format) |
| `vit_nvfp4_bs_064.onnx` | NVFP4 | 48 MB | NVIDIA 4-bit floating point |

### Tools Used
- **NVIDIA TensorRT**: Deep learning inference optimizer
- **Nsight Systems (nsys)**: System-wide GPU activity profiler
- **Nsight Compute (ncu)**: Low-level GPU kernel profiler
- **ONNX Runtime**: Cross-platform inference engine

---

## TensorRT Deep Dive

### TensorRT vs trtexec

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NVIDIA TensorRT SDK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│   │  C++ API         │    │  Python API      │    │  trtexec         │      │
│   │  (libnvinfer)    │    │  (tensorrt pkg)  │    │  (CLI tool)      │      │
│   └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘      │
│            │                       │                       │                 │
│            └───────────────────────┴───────────────────────┘                 │
│                                    │                                         │
│                         ┌──────────▼──────────┐                              │
│                         │   TensorRT Runtime  │                              │
│                         │   (Core Engine)     │                              │
│                         └──────────┬──────────┘                              │
│                                    │                                         │
│                         ┌──────────▼──────────┐                              │
│                         │   CUDA Kernels      │                              │
│                         │   cuDNN, cuBLAS     │                              │
│                         └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | What It Is | When to Use |
|-----------|-----------|-------------|
| **TensorRT** | The optimization SDK/library | Building production inference systems |
| **trtexec** | Command-line tool that wraps TensorRT | Quick benchmarking, profiling, testing |
| **Python API** | Python bindings for TensorRT | Integration with Python ML pipelines |
| **C++ API** | Native TensorRT interface | Maximum performance, production deployment |

### How TensorRT Accelerates Models

#### 1. Layer/Kernel Fusion

```
BEFORE (ONNX - 5 separate CUDA kernel launches):
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ Conv │ → │ BN   │ → │ ReLU │ → │ Conv │ → │ ReLU │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘
   ↓          ↓          ↓          ↓          ↓
 kernel    kernel     kernel     kernel     kernel
 launch    launch     launch     launch     launch
   │          │          │          │          │
   └──────────┴──────────┴──────────┴──────────┘
              Memory transfers between each!

AFTER (TensorRT - 2 fused kernel launches):
┌────────────────────────┐   ┌────────────────────┐
│  Conv + BN + ReLU      │ → │  Conv + ReLU       │
│  (fused kernel)        │   │  (fused kernel)    │
└────────────────────────┘   └────────────────────┘
          ↓                            ↓
     1 kernel launch              1 kernel launch
          │                            │
          └────────────────────────────┘
              No intermediate memory!
```

**Why fusion helps:**
- Fewer kernel launches (each launch has ~5-10μs overhead)
- No intermediate memory read/writes
- Better GPU occupancy

#### 2. Precision Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                    Precision Options                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FP32 (default)     FP16              INT8           FP4/FP8   │
│   ┌───────────┐     ┌───────────┐    ┌───────────┐  ┌────────┐ │
│   │ 32 bits   │     │ 16 bits   │    │ 8 bits    │  │ 4 bits │ │
│   │ per value │     │ per value │    │ per value │  │ per val│ │
│   └───────────┘     └───────────┘    └───────────┘  └────────┘ │
│                                                                 │
│   Speed: 1x          Speed: 2x       Speed: 4x      Speed: 8x  │
│   Accuracy: 100%     Accuracy: ~99%  Accuracy: ~97% Varies     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Kernel Auto-Tuning

For each layer, TensorRT benchmarks multiple kernel implementations:

```
MatMul Operation:
┌─────────────────────────────────────────────────────────────────┐
│  Implementation 1: cuBLAS GEMM          → Benchmark: 0.45ms    │
│  Implementation 2: cuBLAS GEMM (tiled)  → Benchmark: 0.38ms    │
│  Implementation 3: Custom fused kernel  → Benchmark: 0.31ms    │
│  Implementation 4: Tensor Core kernel   → Benchmark: 0.28ms  ✓ │
└─────────────────────────────────────────────────────────────────┘
                                                    ↑
                                          TensorRT picks fastest
```

This is why **engine build takes minutes** - it's benchmarking hundreds of kernel variants!

#### 4. Memory Optimization

```
BEFORE:                              AFTER:
┌─────────┐                         ┌─────────┐
│ Layer 1 │ → Tensor A (100MB)      │ Layer 1 │ → Buffer 1 (100MB)
│ Layer 2 │ → Tensor B (100MB)      │ Layer 2 │ → Buffer 1 (reused!)
│ Layer 3 │ → Tensor C (100MB)      │ Layer 3 │ → Buffer 2 (100MB)
│ Layer 4 │ → Tensor D (100MB)      │ Layer 4 │ → Buffer 1 (reused!)
└─────────┘                         └─────────┘
Total: 400MB                        Total: 200MB
```

---

## What is a TensorRT Engine?

### The Analogy: Source Code vs Compiled Binary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   SOFTWARE WORLD                        DEEP LEARNING WORLD                 │
│                                                                             │
│   ┌──────────────┐                      ┌──────────────┐                   │
│   │   C++ Code   │                      │  ONNX Model  │                   │
│   │  (portable)  │                      │  (portable)  │                   │
│   └──────┬───────┘                      └──────┬───────┘                   │
│          │                                     │                            │
│          │ compile                             │ TensorRT build             │
│          │ (gcc/clang)                         │ (optimization)             │
│          ▼                                     ▼                            │
│   ┌──────────────┐                      ┌──────────────┐                   │
│   │    Binary    │                      │   Engine     │                   │
│   │   (.exe)     │                      │  (.engine)   │                   │
│   │ (CPU-specific)                      │ (GPU-specific)                   │
│   └──────────────┘                      └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Comparison

| Component | What It Is | Contains | Portable? |
|-----------|-----------|----------|-----------|
| **ONNX Model** | Model definition | Weights + graph structure | ✅ Yes - runs anywhere |
| **TensorRT Engine** | Compiled model | Optimized CUDA kernels + weights | ❌ No - GPU-specific |
| **TensorRT Runtime** | Execution library | Code to run engines | ✅ Yes (with matching version) |

### What's Inside an ONNX Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONNX FILE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Graph Definition (generic):                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Input → Conv2D → BatchNorm → ReLU → Conv2D → Output    │   │
│   │         (describes WHAT to compute)                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Weights (raw):                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  conv1.weight: [64, 3, 7, 7] floats                     │   │
│   │  conv1.bias: [64] floats                                │   │
│   │  bn1.weight: [64] floats                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Metadata: opset_version, ir_version, producer                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Size: ~166 MB (vit_fp16)
```

### What's Inside a TensorRT Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                      TensorRT ENGINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Compiled CUDA Kernels (GPU-specific):                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  kernel_0: fused_conv_bn_relu_h100_sm90_fp16            │   │
│   │  kernel_1: attention_flash_h100_sm90_fp16               │   │
│   │  kernel_2: gemm_tensor_core_h100_sm90_fp16              │   │
│   │  (HOW to compute, optimized for YOUR specific GPU)      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Optimized Weights: reformatted for tensor cores               │
│   Execution Plan: memory allocation, kernel launch order        │
│   Device Info: Built for NVIDIA H100 (SM 9.0)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Size: ~50-300 MB (varies based on optimizations)
```

### The Engine Build Process

```
┌──────────────┐
│  ONNX Model  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. PARSE                                                        │
│     - Read ONNX graph                                            │
│     - Validate operations                                        │
│     - Map to TensorRT layers                                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. OPTIMIZE                                                     │
│     - Layer fusion (Conv+BN+ReLU → single kernel)               │
│     - Precision selection (FP32 → FP16/INT8)                    │
│     - Dead code elimination                                      │
│     - Constant folding                                           │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. KERNEL AUTO-TUNE (why build takes minutes!)                 │
│     - Try multiple kernel implementations                        │
│     - Benchmark each on YOUR specific GPU                        │
│     - Select fastest for each layer                              │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. SERIALIZE                                                    │
│     - Pack everything into .engine file                         │
│     - Store selected kernels + weights + execution plan         │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   ENGINE     │ ← Ready to run, no more optimization needed
└──────────────┘
```

### Why Engines Are GPU-Specific

```
Same ONNX Model → Different Engines for Different GPUs:

┌──────────────┐
│  ONNX Model  │
│  (portable)  │
└──────┬───────┘
       │
       ├────────────────────┬────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Build on H100│    │ Build on A100│    │Build on 4090 │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Engine (H100)│    │ Engine (A100)│    │ Engine (4090)│
│  - SM 9.0    │    │  - SM 8.0    │    │  - SM 8.9    │
│  - 80GB HBM3 │    │  - 80GB HBM2e│    │  - 24GB GDDR6│
└──────────────┘    └──────────────┘    └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   ✗ Won't run          ✗ Won't run          ✗ Won't run
   on A100!             on H100!             on H100!
```

---

## Complete Profiling Workflow

```
                              PROFILING WORKFLOW
                              
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   1. MODEL EXPORT (done previously)                         │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐                │
    │   │ PyTorch │ →  │  ONNX   │    │ Quantize│                │
    │   │  Model  │    │ Export  │ →  │ (FP16,  │                │
    │   └─────────┘    └─────────┘    │ MXFP8,  │                │
    │                                 │ NVFP4)  │                │
    │                                 └────┬────┘                │
    │                                      │                      │
    │   2. ENGINE BUILD (TensorRT)         ▼                      │
    │   ┌─────────────────────────────────────────┐               │
    │   │  vit_fp16.onnx ──┐                      │               │
    │   │  vit_mxfp8.onnx ─┼─→ TensorRT Builder   │               │
    │   │  vit_nvfp4.onnx ─┘   (optimize+compile) │               │
    │   │                            │            │               │
    │   │                   ┌────────┴────────┐   │               │
    │   │                   ▼        ▼        ▼   │               │
    │   │              .engine  .engine  .engine  │               │
    │   └─────────────────────────────────────────┘               │
    │                                                              │
    │   3. PROFILED INFERENCE (Nsight Systems)                    │
    │   ┌─────────────────────────────────────────┐               │
    │   │  nsys wraps inference execution         │               │
    │   │  ┌─────────────────────────────────┐    │               │
    │   │  │ Load Engine                     │    │               │
    │   │  │ Warm-up (50 iterations)         │    │               │
    │   │  │ Benchmark (100 iterations) ◀────┼────┼── Measured    │
    │   │  │ Record all GPU activity    ◀────┼────┼── Profiled    │
    │   │  └─────────────────────────────────┘    │               │
    │   └─────────────────────────────────────────┘               │
    │                          │                                   │
    │                          ▼                                   │
    │   4. OUTPUTS                                                │
    │   ┌─────────────────────────────────────────┐               │
    │   │  profile.nsys-rep  → Open in Nsight GUI │               │
    │   │  metrics.json      → Latency/Throughput │               │
    │   │  COMPARISON.txt    → Summary report     │               │
    │   └─────────────────────────────────────────┘               │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

### Nsight Systems Timeline View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Nsight Systems Timeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ CPU Thread  ████░░░░████░░░░████░░░░████░░░░████░░░░                       │
│             │       │       │       │       │                               │
│             ▼       ▼       ▼       ▼       ▼                               │
│ CUDA API    ●       ●       ●       ●       ●  (kernel launches)           │
│             │       │       │       │       │                               │
│             ▼       ▼       ▼       ▼       ▼                               │
│ GPU Stream  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (kernel execution)        │
│                                                                             │
│ Memory      ──▲──▼────────────────────────▲──▼──  (HtoD, DtoH transfers)   │
│                                                                             │
│ Time ──────────────────────────────────────────────────────────────▶       │
│             0ms    2ms    4ms    6ms    8ms   10ms                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Gets Measured in Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Breakdown                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │ Input Copy  │  HtoD: Host memory → GPU memory (~0.1ms)      │
│  └──────┬──────┘                                                │
│         ▼                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Kernel 1   │→ │  Kernel 2   │→ │  Kernel 3   │  ...        │
│  │  (Attention)│  │  (FFN)      │  │  (LayerNorm)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                                    │                  │
│         └────────────────────────────────────┘                  │
│                    GPU Compute (~7ms for FP16+TRT)              │
│         ┌────────────────────────────────────┐                  │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │ Output Copy │  DtoH: GPU memory → Host memory (~0.05ms)     │
│  └─────────────┘                                                │
│                                                                 │
│  Total Latency = Input + Compute + Output                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
profiling/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── models/                             # ONNX models
│   ├── vit_fp16_bs_064.onnx           # FP16 model (166 MB)
│   ├── vit_mxfp8_bs_064.onnx          # MXFP8 model (87 MB)
│   └── vit_nvfp4_bs_064.onnx          # NVFP4 model (48 MB)
│
├── scripts/                            # Profiling scripts
│   ├── run_profiling.sh               # Main profiling script (nsys + ncu + benchmark)
│   └── setup.sh                       # Environment setup
│
├── src/                                # Python source code
│   ├── __init__.py
│   ├── benchmark.py                   # ONNX benchmarking utilities
│   ├── profiler.py                    # NCU profiling wrapper
│   ├── analyzer.py                    # Result analysis
│   ├── compare.py                     # Result comparison utilities
│   ├── compare_all.py                 # Multi-result comparison
│   └── visualizer.py                  # Generate charts/plots
│
├── configs/                            # Configuration files
│   └── profiling_config.yaml          # Centralized profiling configuration
│
├── tools/                              # External tool wrappers
│   ├── nsys_wrapper.sh                # Nsight Systems wrapper
│   ├── ncu_wrapper.sh                 # Nsight Compute wrapper
│   ├── verify_cuda_setup.sh           # CUDA environment verification
│   └── transfer_package.sh            # File transfer utility
│
├── results/                            # Profiling outputs (generated)
│   ├── nsight-systems/                # Nsight Systems results (.nsys-rep)
│   ├── nsight-compute/                # Nsight Compute results (.ncu-rep)
│   ├── benchmark/                     # Benchmark results (JSON)
│   └── PROFILING_REPORT_*.txt         # Summary reports
│
├── engines/                            # TensorRT engines (generated)
│   └── *.engine                       # Compiled TRT engines
│
└── logs/                               # Execution logs (generated)
    └── profiling_*.log
```

---

## Usage

### Prerequisites

- **Docker** with NVIDIA Container Toolkit
- **NVIDIA GPU** with driver installed
- **Container**: `nvcr.io/nvidia/pytorch:25.06-py3` (auto-pulled)

```bash
# Setup environment (pulls container, verifies GPU)
./scripts/setup.sh

# Container includes:
#   - TensorRT 10.11
#   - ModelOpt 0.29 (for MXFP8 support)
#   - Nsight Systems (nsys)
#   - Nsight Compute (ncu)
```

### Run Profiling

All profiling uses Docker containers - no local installation needed.

```bash
# Full profiling (Nsight Systems + Nsight Compute + Benchmark)
./scripts/run_profiling.sh

# Nsight Systems only (GPU timeline, ~10 min)
./scripts/run_profiling.sh --nsys

# Nsight Compute only (kernel metrics, ~30 min - slowest)
./scripts/run_profiling.sh --ncu

# Benchmark only (no profiling overhead, ~5 min - fastest)
./scripts/run_profiling.sh --benchmark

# Build TensorRT engines only
./scripts/run_profiling.sh --build
```

### View Results

```bash
# Text report
cat results/nsight-systems/COMPARISON_REPORT_*.txt

# Open Nsight Systems GUI (on local machine with GUI)
nsys-ui results/nsight-systems/vit_fp16_trt_*/profile.nsys-rep
```

---

## Profiling Your Own Models

### Exporting Your Model to ONNX

If you have a PyTorch model, export it to ONNX:

```python
import torch
import torch.onnx

# Load your model
model = YourModel()
model.eval()

# Create dummy input matching your model's expected input
batch_size = 64
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/your_model_fp16.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### Quantizing to NVFP4

Use NVIDIA ModelOpt to quantize your ONNX model:

```python
# Inside the container or with modelopt installed
import modelopt.onnx.quantization as moq

# Quantize to NVFP4
moq.quantize(
    onnx_path="models/your_model_fp16.onnx",
    output_path="models/your_model_nvfp4.onnx",
    quantize_mode="nvfp4_awq_clip",
)
```

Or use the container:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python3 -c "
import modelopt.onnx.quantization as moq
moq.quantize(
    '/workspace/models/your_model_fp16.onnx',
    '/workspace/models/your_model_nvfp4.onnx',
    quantize_mode='nvfp4_awq_clip'
)
"
```

### Updating the Profiling Script

1. **Edit model list** in `scripts/run_profiling.sh`:

```bash
declare -a MODELS=(
    "your_model_fp16.onnx:fp16"
    "your_model_nvfp4.onnx:nvfp4"
)
```

2. **Adjust input shapes** if needed (in trtexec commands):

```bash
# For dynamic shapes, add to trtexec:
--minShapes=input:1x3x224x224 \
--optShapes=input:64x3x224x224 \
--maxShapes=input:128x3x224x224
```

### Expected Results by Model Size

| Model Size | FP16 Engine | NVFP4 Engine | Build Time |
|------------|-------------|--------------|------------|
| ~100M params | ~400 MB | ~150 MB | ~2 min |
| ~300M params | ~1.2 GB | ~400 MB | ~5 min |
| ~1B params | ~4 GB | ~1.5 GB | ~15 min |

---

## Results

### Example: FP16 CUDA vs TensorRT

| Configuration | Mean Latency | P95 Latency | Throughput |
|--------------|-------------|-------------|------------|
| FP16 + CUDA | 2793.07 ms | 3913.97 ms | 22.91 imgs/sec |
| FP16 + TensorRT | 7.44 ms | 7.48 ms | 8,607 imgs/sec |

**TensorRT Speedup: 375x** 🚀

### Why Such Large Speedup?

| Factor | CUDA Provider | TensorRT |
|--------|--------------|----------|
| Kernel fusion | None | ✅ Aggressive fusion |
| Precision | FP32 compute | FP16 Tensor Cores |
| Memory | Naive allocation | Optimized reuse |
| Kernels | Generic cuDNN | Auto-tuned for GPU |

---

## Known Issues

### MXFP8/NVFP4 Model Compatibility

The MXFP8 and NVFP4 models contain TensorRT-specific custom operators:
- `trt` domain operators
- `trt.plugins` domain operators

These require **native TensorRT** (not through ONNX Runtime) to execute:

```
Standard ONNX Model (FP16):     → Works with ONNX Runtime ✓
TRT-Specific Model (MXFP8):     → Requires native TensorRT
```

**Solution**: Use `trtexec` directly or TensorRT Python API.

---

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

---

## License

Internal NVIDIA project for performance profiling and optimization research.
