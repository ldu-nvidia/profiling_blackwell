# ViT Profiling - TensorRT on Blackwell

Profile Vision Transformer (ViT) models with different precision formats on NVIDIA Blackwell GPUs.

## Results (Batch Size 64)

### TensorRT 10.14.1 (`tensorrt:25.11-py3`)

| Precision | GPU Compute | Speedup vs FP16 |
|-----------|-------------|-----------------|
| FP16 | 8.23 ms | 1.00x (baseline) |
| MXFP8 | 6.55 ms | **1.26x** |
| NVFP4 | 5.14 ms | **1.60x** |

### TensorRT 10.16.0 (Pre-release, custom container)

| Precision | GPU Compute | Speedup vs FP16 |
|-----------|-------------|-----------------|
| FP16 | 7.90 ms | 1.00x (baseline) |
| MXFP8 | 6.52 ms | **1.21x** |
| NVFP4 | 5.16 ms | **1.53x** |

### Comparison: TRT 10.14 vs TRT 10.16

| Precision | TRT 10.14 | TRT 10.16 | Absolute Improvement |
|-----------|-----------|-----------|----------------------|
| FP16 | 8.23 ms | **7.90 ms** | 🚀 4% faster |
| MXFP8 | 6.55 ms | **6.52 ms** | ~same |
| NVFP4 | 5.14 ms | 5.16 ms | ~same |

**Key Finding:** TRT 10.16 has improved FP16 kernels for Blackwell (SM120), resulting in a faster baseline. Quantized models show similar absolute performance but lower relative speedup.

---

## Quick Start

### Option A: TensorRT 10.14.1 (Official NGC Container)

```bash
docker pull nvcr.io/nvidia/tensorrt:25.11-py3
```

### Option B: TensorRT 10.16.0 (Custom Container)

Requires `TensorRT-10.16.0.12.Linux.x86_64-gnu.cuda-13.1.tar.gz` from NVIDIA Developer.

---

## Build and Profile Engines

### Using TensorRT 10.14.1

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

### Using TensorRT 10.16.0

```bash
cd /home/ldu/repos/profiling_blackwell

# FP16 (baseline)
docker run --rm --gpus all -v $(pwd):/workspace tensorrt-10.16:latest \
    trtexec --onnx=/workspace/models/vit_fp16_bs_064.onnx \
            --saveEngine=/workspace/engines/fp16_trt1016.engine \
            --fp16

# MXFP8 (1.21x speedup)
docker run --rm --gpus all -v $(pwd):/workspace tensorrt-10.16:latest \
    trtexec --onnx=/workspace/models/vit_mxfp8_bs_064.onnx \
            --saveEngine=/workspace/engines/mxfp8_trt1016.engine \
            --fp16 --stronglyTyped

# NVFP4 (1.53x speedup)
docker run --rm --gpus all -v $(pwd):/workspace tensorrt-10.16:latest \
    trtexec --onnx=/workspace/models/vit_nvfp4_bs_064.onnx \
            --saveEngine=/workspace/engines/nvfp4_trt1016.engine \
            --fp16 --stronglyTyped
```

### Profile Any Engine

```bash
# Replace CONTAINER with tensorrt:25.11-py3 or tensorrt-10.16:latest
docker run --rm --gpus all -v $(pwd):/workspace CONTAINER \
    trtexec --loadEngine=/workspace/engines/ENGINE_FILE.engine \
            --warmUp=500 --iterations=100
```

---

## Profiling Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| GPU | NVIDIA RTX PRO 6000 | Blackwell architecture (SM120) |
| Batch Size | 64 | Static batch |
| Warmup | 500 iterations | Excluded from timing |
| Benchmark | 100 iterations | Timed runs for latency measurement |
| CUDA Graph | Enabled | Reduces kernel launch overhead |
| Data Transfer | Excluded | GPU compute time only (no H2D/D2H) |

**Requirements:**
- NVIDIA Blackwell GPU (RTX PRO 6000, B100, B200)
- Docker with NVIDIA Container Toolkit

---

## Future Optimization

### Top 4 Optimization Opportunities

| Rank | Optimization | Potential Speedup | Architecture Change Required | Effort |
|------|--------------|-------------------|------------------------------|--------|
| **1** | **2:4 Structured Sparsity** | +50-100% | ✅ Yes - Requires retraining with sparsity-aware fine-tuning using ModelOpt | High |
| **2** | **Attention Quantization** | +20-40% | ✅ Yes - Sequence length must be divisible by 32 (MXFP8) or 16 (NVFP4); attention tensors must be 3D | High |
| **3** | **Flash Attention** | +20-30% | ✅ Yes - Must use `scaled_dot_product_attention` or compatible implementation for TRT fusion | Medium |
| **4** | **Increase Batch Size** | +10-20% | ❌ No - Can apply directly to existing ONNX with dynamic shapes | Low |

### Combined Potential Speedup

| Configuration | Cumulative Speedup |
|---------------|-------------------|
| Current (NVFP4 linear only) | **1.60x** |
| + Batch size increase | ~1.80x |
| + Flash Attention | ~2.20x |
| + Attention Quantization | ~2.80x |
| + 2:4 Sparsity | **~3.50-4.00x** |

---

### Architecture Design Rules for Attention Quantization

To enable native FP4/FP8 attention quantization without graph surgery, the ViT model must be designed with specific dimension constraints.

#### Rule 1: Sequence Length Must Be Divisible by Block Size

| Precision | Block Size | Valid Sequence Lengths |
|-----------|------------|------------------------|
| MXFP8 | 32 | 64, 128, 192, **256**, 320, 384, 512 |
| NVFP4 | 16 | 64, 128, 192, 208, **256**, 320, 384, 512 |

```python
# ❌ BAD: 197 (196 patches + 1 class token)
self.seq_len = (224 // 16) ** 2 + 1  # = 197 (NOT divisible!)

# ✅ GOOD: 256 (no class token)
self.seq_len = (256 // 16) ** 2      # = 256 (divisible by 32 and 16)
```

#### Rule 2: Head Dimension Must Be Divisible by Block Size

| Precision | Block Size | Valid Head Dimensions |
|-----------|------------|----------------------|
| MXFP8 | 32 | 32, 64, 96, 128 |
| NVFP4 | 16 | 16, 32, 48, 64, 80, 96, 128 |

#### Rule 3: Export Attention Tensors as 3D (Not 4D)

TensorRT's quantization ops only support 2D or 3D tensors.

```python
# ❌ BAD: 4D attention tensors [batch, heads, seq_len, head_dim]
q = q.view(B, self.num_heads, S, self.head_dim)
attn = torch.matmul(q, k.transpose(-2, -1))

# ✅ GOOD: 3D attention tensors [batch * heads, seq_len, head_dim]
q = q.view(B * self.num_heads, S, self.head_dim)
attn = torch.bmm(q, k.transpose(-2, -1))
```

#### Rule 4: Avoid Class Token (Or Pad Sequence)

```python
# ❌ BAD: Class token breaks divisibility (196 + 1 = 197)

# ✅ OPTION A: No class token (use global average pooling)
x = self.transformer(x)
x = x.mean(dim=1)  # Global average pooling

# ✅ OPTION B: Pad sequence to valid length (197 → 256)
```

#### Rule 5: Use Compatible Image/Patch Size Combinations

| Image Size | Patch Size | Num Patches | Divisible by 32? |
|------------|------------|-------------|------------------|
| 224 | 16 | 196 (+1=197) | ❌ |
| **256** | **16** | **256** | ✅ |
| 384 | 24 | 256 | ✅ |

#### Quick Reference Checklist

```
□ Sequence length divisible by 32 (MXFP8) or 16 (NVFP4)
□ Head dimension divisible by 32 (MXFP8) or 16 (NVFP4)
□ Attention tensors exported as 3D [B*H, S, D]
□ No class token OR pad sequence to valid length
□ Image/patch size produces valid num_patches
□ Use ModelOpt for quantization-aware export
□ Use torch.bmm() instead of torch.matmul() for attention
```

#### Recommended ViT Configuration

```python
class QuantizationFriendlyViT(nn.Module):
    def __init__(self):
        self.image_size = 256
        self.patch_size = 16
        self.num_patches = 256      # (256/16)^2 = 256
        self.seq_len = 256          # No class token
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = 64          # 768/12 = 64 (divisible by 32)
        self.use_cls_token = False  # Use global avg pooling instead
```
