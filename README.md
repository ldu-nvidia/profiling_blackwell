# ViT Profiling - TensorRT on Blackwell

Profile Vision Transformer (ViT) models with different precision formats on NVIDIA Blackwell GPUs.

## Results (Batch Size 64)

| Precision | GPU Compute | Speedup vs FP16 |
|-----------|-------------|-----------------|
| FP16 | 8.23 ms | 1.00x (baseline) |
| MXFP8 | 6.55 ms | **1.26x** |
| NVFP4 | 5.14 ms | **1.60x** |

---

## Quick Start

### Step 1: Pull Container

```bash
docker pull nvcr.io/nvidia/tensorrt:25.11-py3
```

### Step 2: Build Engines

```bash
cd /home/ldu/repos/profiling_blackwell

# FP16 (baseline)
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --onnx=/workspace/models/vit_fp16_bs_064.onnx \
            --saveEngine=/workspace/engines/vit_fp16.engine \
            --fp16

# MXFP8 (1.26x speedup)
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --onnx=/workspace/models/vit_mxfp8_bs_064.onnx \
            --saveEngine=/workspace/engines/vit_mxfp8.engine \
            --fp16 --stronglyTyped

# NVFP4 (1.60x speedup)
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --onnx=/workspace/models/vit_nvfp4_bs_064.onnx \
            --saveEngine=/workspace/engines/vit_nvfp4.engine \
            --fp16 --stronglyTyped
```

### Step 3: Profile Engines

```bash
# Profile FP16
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --loadEngine=/workspace/engines/vit_fp16.engine \
            --warmUp=500 --iterations=100

# Profile MXFP8
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --loadEngine=/workspace/engines/vit_mxfp8.engine \
            --warmUp=500 --iterations=100

# Profile NVFP4
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:25.11-py3 \
    trtexec --loadEngine=/workspace/engines/vit_nvfp4.engine \
            --warmUp=500 --iterations=100
```

---

## Project Structure

```
profiling_blackwell/
├── models/                      # ONNX models
│   ├── vit_fp16_bs_064.onnx     # FP16 baseline (173 MB)
│   ├── vit_mxfp8_bs_064.onnx    # MXFP8 quantized (90 MB)
│   └── vit_nvfp4_bs_064.onnx    # NVFP4 quantized (50 MB)
├── engines/                     # Built TensorRT engines
├── output/                      # Logs and profiling results
├── docs/                        # Documentation and reports
├── scripts/                     # Automation scripts
└── configs/                     # Configuration files
```

## Requirements

- NVIDIA Blackwell GPU (RTX PRO 6000, B100, B200)
- Docker with NVIDIA Container Toolkit
- Container: `nvcr.io/nvidia/tensorrt:25.11-py3` (TensorRT 10.14.1)

## Key Flags

| Flag | Purpose |
|------|---------|
| `--fp16` | Enable FP16 precision |
| `--stronglyTyped` | Required for quantized models (MXFP8/NVFP4) to preserve quantization |
| `--warmUp=500` | Warmup iterations before timing |
| `--iterations=100` | Number of timed iterations |
