#!/usr/bin/env python3
"""
Create FP8 Quantized ViT ONNX Model

This script:
1. Creates a clean FP16 ViT ONNX model
2. Uses ModelOpt's ONNX-level FP8 quantization 
3. Produces a TensorRT-compatible FP8 model

The result is a model with FP8 activations that TensorRT can accelerate.

Usage:
    docker run --gpus all --rm -v $(pwd):/workspace \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        nvcr.io/nvidia/pytorch:25.06-py3 \
        bash -c "pip install huggingface_hub 'transformers<4.45' onnx_graphsurgeon onnxruntime onnxconverter_common --quiet && \
                 python /workspace/scripts/create_vit_fp8.py"
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='Create FP8 Quantized ViT ONNX Model')
    parser.add_argument('--output-fp16', type=str, default='/workspace/models/vit_fp16_custom.onnx')
    parser.add_argument('--output-fp8', type=str, default='/workspace/models/vit_fp8_bs_064.onnx')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Create FP16 ViT model
    print("\n" + "="*60)
    print("Step 1: Creating ViT-Base/16 Model")
    print("="*60)
    
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=1000,
        embed_dim=768, depth=12, num_heads=12
    ).to(device).half()  # FP16
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Step 2: Export to FP16 ONNX
    print("\n" + "="*60)
    print("Step 2: Exporting to FP16 ONNX")
    print("="*60)
    
    os.makedirs(os.path.dirname(args.output_fp16), exist_ok=True)
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device, dtype=torch.float16)
    
    torch.onnx.export(
        model, dummy_input, args.output_fp16,
        input_names=['input'], output_names=['output'],
        opset_version=17, do_constant_folding=True
        # Fixed batch size - no dynamic_axes
    )
    
    print(f"FP16 ONNX exported: {args.output_fp16}")
    print(f"File size: {os.path.getsize(args.output_fp16) / 1024 / 1024:.1f} MB")
    
    # Step 3: Apply FP8 quantization using ModelOpt ONNX quantization
    print("\n" + "="*60)
    print("Step 3: Applying FP8 Quantization")
    print("="*60)
    
    try:
        import modelopt.onnx.quantization as moq
        
        # Create calibration data - must match batch size
        print("Generating calibration data...")
        calib_data = np.random.randn(args.batch_size, 3, 224, 224).astype(np.float16)
        
        print("Running FP8 quantization...")
        moq.quantize(
            onnx_path=args.output_fp16,
            quantize_mode='fp8',
            calibration_data={'input': calib_data},
            output_path=args.output_fp8,
            verbose=True
        )
        
        print(f"\n✅ FP8 ONNX model created: {args.output_fp8}")
        print(f"File size: {os.path.getsize(args.output_fp8) / 1024 / 1024:.1f} MB")
        
        # Verify
        import onnx
        model = onnx.load(args.output_fp8)
        from collections import Counter
        ops = Counter(n.op_type for n in model.graph.node)
        
        print("\n=== Quantization Ops ===")
        for op, count in sorted(ops.items()):
            if 'Quant' in op or 'FP' in op.upper():
                print(f"  {op}: {count}")
        
    except ImportError as e:
        print(f"ModelOpt ONNX quantization not available: {e}")
        print("Please install: pip install onnx_graphsurgeon onnxruntime onnxconverter_common")
        return
    except Exception as e:
        print(f"FP8 quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("FP8 Model Creation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

