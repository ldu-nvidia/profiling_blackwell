# Vision Transformer (ViT) Performance Analysis
## FP16 vs NVFP4 Quantization on NVIDIA Blackwell GPU

**Date:** December 3, 2025  
**GPU:** NVIDIA RTX PRO 6000 Blackwell Workstation Edition  
**Model:** ViT-Base, Batch Size 64  
**TensorRT Version:** 10.11.0  

---

## 1. Executive Summary

| Metric | FP16 | NVFP4 | Improvement |
|--------|------|-------|-------------|
| **Mean Latency** | 10.4 ms | 7.3 ms | 1.42x faster |
| **Throughput** | 95.8 qps | 136.4 qps | 1.42x higher |
| **Engine Size** | 169 MB | 52 MB | 3.25x smaller |

**Key Finding:** The observed 1.42x speedup is significantly below the theoretical 4x improvement expected from FP4 quantization. This report explains why.

---

## 2. Benchmark Configuration

```yaml
Container: nvcr.io/nvidia/pytorch:25.06-py3
Warmup Iterations: 50
Benchmark Iterations: 100
CUDA Graph: Enabled
Data Transfers: Disabled (GPU-only timing)
```

### Model Specifications

| Model | Precision | ONNX Size | Engine Size |
|-------|-----------|-----------|-------------|
| vit_fp16_bs_064.onnx | FP16 | 173 MB | 169 MB |
| vit_nvfp4_bs_064.onnx | NVFP4 (4-bit weights) | 50 MB | 52 MB |

---

## 3. Kernel-Level Performance Breakdown

### 3.1 FP16 Model - GPU Kernel Distribution

```
Total GPU Kernel Time: ~7.6 ms

┌─────────────────────────────────────────────────────────────────┐
│ FP16 GEMM (Linear Projections)           ████████████████ 84.2% │
│ Multi-Head Attention                     ███               7.1% │
│ LayerNorm                                ██                4.9% │
│ Other                                    █                 3.8% │
└─────────────────────────────────────────────────────────────────┘
```

**Top Kernels:**

| Rank | Time % | Kernel | Instances | Avg (µs) |
|------|--------|--------|-----------|----------|
| 1 | 29.9% | `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize256x128x32` | 12 | 188.3 |
| 2 | 26.8% | `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize256x128x32` | 12 | 168.4 |
| 3 | 20.0% | `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize128x256x32` | 12 | 126.0 |
| 4 | 7.5% | `sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize160x128x32` | 12 | 47.5 |
| 5 | 7.1% | `_gemm_mha_v2` (Attention) | 12 | 44.5 |
| 6 | 4.9% | `__myl_AddCastMeanSubMul...` (LayerNorm) | 22 | 16.8 |

### 3.2 NVFP4 Model - GPU Kernel Distribution

```
Total GPU Kernel Time: ~5.5 ms

┌─────────────────────────────────────────────────────────────────┐
│ FP16 GEMM (Still present!)               ██████████████   26.5% │
│ FP4 Block-Scaled GEMM                    █████████████    25.0% │
│ Multi-Head Attention                     █████             9.7% │
│ GELU + Dequantization                    █████             9.7% │
│ LayerNorm                                ███               6.9% │
│ Format Conversion                        ██                5.5% │
│ Other                                    ████             16.7% │
└─────────────────────────────────────────────────────────────────┘
```

**Top Kernels:**

| Rank | Time % | Kernel | Instances | Avg (µs) |
|------|--------|--------|-----------|----------|
| 1 | 26.5% | `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n` (FP16!) | 12 | 120.8 |
| 2 | **25.0%** | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1` (FP4) | 36 | 38.1 |
| 3 | 9.7% | `__myl_DivCastErfCastAddMulMulResh` (GELU+dequant) | 12 | 44.4 |
| 4 | 9.7% | `_gemm_mha_v2` (Attention) | 12 | 44.0 |
| 5 | 6.9% | `__myl_AddCastMeanSubMul...` (LayerNorm) | 22 | 17.2 |
| 6 | 2.9% | `__myl_ReshReshCastMulReplCast...` (Conversion) | 12 | 13.4 |

---

## 4. Root Cause Analysis

### 4.1 Why Only 1.42x Instead of 4x?

```
                    ┌──────────────────────────────────────┐
                    │     SPEEDUP LIMITING FACTORS         │
                    └──────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   PARTIAL     │           │  ATTENTION    │           │ DEQUANT       │
│ QUANTIZATION  │           │  NOT FP4      │           │ OVERHEAD      │
│               │           │               │           │               │
│ Only 25% of   │           │ Softmax needs │           │ ~5% time in   │
│ compute uses  │           │ FP16+ for     │           │ format        │
│ FP4 kernels   │           │ stability     │           │ conversion    │
└───────────────┘           └───────────────┘           └───────────────┘
     MAJOR                       MAJOR                      MINOR
```

### 4.2 Quantization Coverage Analysis

**FP4 Quantization Coverage in NVFP4 Model:**

| Component | Quantized? | Impact |
|-----------|------------|--------|
| Patch Embedding Conv | ❌ FP16 | Minor (1 layer) |
| Q, K, V Projections | ⚠️ Partial | Major |
| Attention Mechanism | ❌ FP16 | Major |
| MLP FC1, FC2 | ✅ FP4 | Beneficial |
| Output Head | ❌ FP16 | Minor (1 layer) |
| LayerNorm | ❌ FP16/32 | Cannot quantize |
| GELU | ❌ FP16 | Cannot quantize |

### 4.3 Attention Mechanism Comparison

```
                     FP16                 NVFP4
                   ─────────           ─────────
_gemm_mha_v2:      533 µs              528 µs
                      │                    │
                      └────────┬───────────┘
                               │
                        Nearly Identical!
                               │
                               ▼
              Attention runs in FP16 in BOTH models
              (Softmax requires numerical precision)
```

---

## 5. Theoretical vs Actual Speedup

### Mathematical Analysis

```
FP16 Breakdown:
├── GEMM operations:     6.36 ms (84%)
└── Non-GEMM operations: 1.24 ms (16%)

If ALL GEMMs were FP4 @ 4x speedup:
├── FP4 GEMM time:       1.59 ms (6.36 ÷ 4)
├── Non-GEMM time:       1.24 ms (unchanged)
├── Total:               2.83 ms
└── Expected Speedup:    2.68x

Actual NVFP4:
├── Only ~48% of GEMMs are FP4
├── FP4 GEMMs:           1.37 ms
├── FP16 GEMMs:          1.45 ms (still present!)
├── Non-GEMM + overhead: 2.68 ms
├── Total:               5.50 ms
└── Actual Speedup:      1.38x
```

### Speedup Ceiling

```
┌────────────────────────────────────────────────────────────────────┐
│                        SPEEDUP POTENTIAL                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Theoretical Max (all compute FP4):         ████████████████  4.0x │
│  If all GEMMs FP4:                          ██████████        2.7x │
│  Current NVFP4 model:                       ██████            1.4x │
│                                                                    │
│  ──────────────────────────────────────────────────────────────    │
│  Limiting factors:                                                 │
│    • Partial quantization       -1.3x potential                    │
│    • Attention in FP16          -0.5x potential                    │
│    • Dequant overhead           -0.1x potential                    │
│    • Memory bound ops           -0.4x potential                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. Memory Analysis

### Engine Size Comparison

| Model | Size | Reduction |
|-------|------|-----------|
| FP16 | 169 MB | Baseline |
| NVFP4 | 52 MB | **3.25x smaller** |

**Note:** Memory reduction (3.25x) exceeds compute speedup (1.42x), indicating the model is not fully compute-bound at batch size 64.

### Memory Bandwidth

```
FP16 Model:
├── H2D Transfer: 192 MB
├── Memset: 346 MB
└── D2H Transfer: 0.13 MB

NVFP4 Model:
├── H2D Transfer: 71 MB  (2.7x less)
├── Memset: 334 MB
└── D2H Transfer: 0.13 MB
```

---

## 7. Recommendations

### 7.1 To Improve Speedup

| Action | Expected Impact | Difficulty |
|--------|-----------------|------------|
| Quantize all linear layers to FP4 | +0.5-1.0x | Medium |
| Use MXFP8 for attention | +0.3-0.5x | High (needs plugin) |
| Increase batch size to 128+ | +0.2-0.4x | Easy |
| Fused dequantization kernels | +0.1x | High |

### 7.2 When FP4 Works Best

- ✅ Large batch sizes (compute-bound)
- ✅ Fully quantized linear layers
- ✅ Models with high GEMM percentage
- ✅ Memory-constrained deployments

### 7.3 When FP16 May Be Preferable

- ❌ Small batch sizes (memory-bound)
- ❌ Accuracy-sensitive applications
- ❌ Models with significant non-GEMM compute

---

## 8. Profiling Artifacts

### Generated Files

```
profiling/
├── reports/
│   └── vit_fp16_vs_nvfp4_analysis.md  (this report)
├── results/
│   ├── nsight-systems/
│   │   ├── vit_fp16_20251203_233040/
│   │   │   ├── profile.nsys-rep
│   │   │   └── profile.sqlite
│   │   └── vit_nvfp4_20251203_233040/
│   │       ├── profile.nsys-rep
│   │       └── profile.sqlite
│   └── benchmark/
│       ├── fp16_20251203_233040.json
│       └── nvfp4_20251203_233040.json
└── engines/
    ├── fp16_20251203_233040.engine
    └── nvfp4_20251203_233040.engine
```

### How to Reproduce

```bash
# Run full profiling
cd /home/ldu/repos/profiling
./scripts/run_profiling.sh

# Generate kernel stats
docker run --rm --gpus all \
  -v $(pwd):/workspace/profiling \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  nsys stats --force-export=true \
  /workspace/profiling/results/nsight-systems/vit_fp16_*/profile.nsys-rep
```

---

## 9. Conclusion

The NVFP4 ViT model achieves a **1.42x speedup** over FP16, significantly below the theoretical 4x improvement. The primary bottlenecks are:

1. **Incomplete quantization** - Only ~25% of compute time uses FP4 kernels
2. **FP16 attention mechanism** - Softmax requires higher precision
3. **Dequantization overhead** - ~5% of time in format conversion

To approach 4x speedup, the model would need comprehensive FP4 quantization of all linear layers and potentially FP8 for attention computations.

---

*Report generated from Nsight Systems profiling data*  
*Analysis performed on NVIDIA RTX PRO 6000 Blackwell Workstation Edition*

