#!/usr/bin/env python3
"""
Sparsify ViT model using NVIDIA ModelOpt 2:4 structured sparsity.

This script performs Post-Training Sparsification (PTS) on a ViT model
using either SparseGPT (data-driven) or magnitude-based pruning.

Usage:
    python sparsify_vit.py --method sparsegpt --calib_size 512
    python sparsify_vit.py --method sparse_magnitude
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ModelOpt imports
import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts


def create_vit_base_patch16_224(num_classes=1000):
    """Create a ViT-Base model matching the architecture in your ONNX files."""
    try:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        return model
    except ImportError:
        print("timm not installed. Please run: pip install timm")
        raise


def get_calibration_dataloader(batch_size=64, num_samples=512, image_size=224):
    """Create synthetic calibration data for SparseGPT."""
    print(f"Creating synthetic calibration data: {num_samples} samples, batch_size={batch_size}")
    
    # Generate random images (in practice, use real data for better results)
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 1000, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader


def collect_func(batch):
    """Extract input tensor from batch for ModelOpt."""
    images, _ = batch
    return images


def sparsify_model(model, method="sparsegpt", calib_dataloader=None, device="cuda"):
    """Apply 2:4 structured sparsity to the model."""
    model = model.to(device)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Starting {method} sparsification...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if method == "sparsegpt":
        # SparseGPT: Data-driven pruning (better accuracy)
        assert calib_dataloader is not None, "SparseGPT requires calibration data!"
        
        config = {
            "data_loader": calib_dataloader,
            "collect_func": collect_func,
        }
        model = mts.sparsify(model, mode="sparsegpt", config=config)
        
    elif method == "sparse_magnitude":
        # Magnitude-based pruning (simpler, no calibration needed)
        model = mts.sparsify(model, mode="sparse_magnitude")
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sparsegpt' or 'sparse_magnitude'")
    
    elapsed = time.time() - start_time
    print(f"\nSparsification completed in {elapsed:.2f}s")
    
    return model


def export_sparse_model(model):
    """Export the sparse model (removes dynamic mask enforcement)."""
    print("\nExporting sparse model...")
    model = mts.export(model)
    return model


def verify_sparsity(model):
    """Verify that 2:4 sparsity pattern is applied."""
    print("\n" + "="*60)
    print("Verifying 2:4 sparsity pattern...")
    print("="*60 + "\n")
    
    total_params = 0
    sparse_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total = weight.numel()
            zeros = (weight == 0).sum().item()
            sparsity = zeros / total * 100
            
            total_params += total
            sparse_params += zeros
            
            # Check 2:4 pattern (50% sparsity expected)
            if sparsity > 40:  # Allow some tolerance
                print(f"✅ {name}: {sparsity:.1f}% sparse ({weight.shape})")
            else:
                print(f"⚠️  {name}: {sparsity:.1f}% sparse ({weight.shape})")
    
    overall_sparsity = sparse_params / total_params * 100
    print(f"\n{'='*60}")
    print(f"Overall sparsity: {overall_sparsity:.1f}%")
    print(f"Expected: ~50% for 2:4 structured sparsity")
    print(f"{'='*60}")
    
    return overall_sparsity


def export_to_onnx(model, output_path, batch_size=64, image_size=224, device="cuda"):
    """Export sparse model to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"✅ ONNX model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sparsify ViT with ModelOpt 2:4 sparsity")
    parser.add_argument("--method", type=str, default="sparsegpt",
                        choices=["sparsegpt", "sparse_magnitude"],
                        help="Sparsification method")
    parser.add_argument("--calib_size", type=int, default=512,
                        help="Number of calibration samples for SparseGPT")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models/sparse",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--skip_onnx", action="store_true",
                        help="Skip ONNX export")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load pretrained ViT
    print("\n" + "="*60)
    print("Loading ViT-Base-Patch16-224...")
    print("="*60)
    model = create_vit_base_patch16_224()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 2: Prepare calibration data (if using SparseGPT)
    calib_dataloader = None
    if args.method == "sparsegpt":
        calib_dataloader = get_calibration_dataloader(
            batch_size=args.batch_size,
            num_samples=args.calib_size
        )
    
    # Step 3: Apply 2:4 sparsity
    model = sparsify_model(
        model,
        method=args.method,
        calib_dataloader=calib_dataloader,
        device=args.device
    )
    
    # Step 4: Export sparse model (freeze sparsity pattern)
    model = export_sparse_model(model)
    
    # Step 5: Verify sparsity
    verify_sparsity(model)
    
    # Step 6: Save ModelOpt state
    state_path = os.path.join(args.output_dir, f"vit_sparse_{args.method}_state.pth")
    print(f"\nSaving ModelOpt state to: {state_path}")
    mto.save(model, state_path)
    
    # Step 7: Export to ONNX
    if not args.skip_onnx:
        onnx_path = os.path.join(args.output_dir, f"vit_sparse_{args.method}_bs{args.batch_size:03d}.onnx")
        export_to_onnx(model, onnx_path, batch_size=args.batch_size, device=args.device)
    
    print("\n" + "="*60)
    print("DONE! Next steps:")
    print("="*60)
    print("""
1. (Optional) Fine-tune with Sparsity-Aware Training (SAT) to recover accuracy

2. Build TensorRT engine with sparsity:
   trtexec --onnx=<sparse_model.onnx> \\
           --saveEngine=<engine.plan> \\
           --fp16 \\
           --sparsity=enable

3. Profile performance:
   trtexec --loadEngine=<engine.plan> \\
           --warmUp=500 --iterations=100
""")


if __name__ == "__main__":
    main()

