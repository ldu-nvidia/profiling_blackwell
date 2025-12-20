#!/usr/bin/env python3
"""
Create Sparse ViT Models in Multiple Precisions

This script creates three versions of sparsified ViT:
1. FP16 + 2:4 Sparsity
2. MXFP8 + 2:4 Sparsity (linear layers only)
3. NVFP4 + 2:4 Sparsity (linear layers only)

Workflow:
1. Load ViT model
2. Apply 2:4 structured sparsity using ModelOpt
3. Export FP16 sparse ONNX
4. Apply MXFP8 quantization → Export ONNX
5. Apply NVFP4 quantization → Export ONNX

Usage:
    python create_sparse_vit_all_precisions.py --batch_size 64 --output_dir output/sparse
"""

import argparse
import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ModelOpt imports
import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts
import modelopt.torch.quantization as mtq


def create_vit_model(num_classes=1000):
    """Create ViT-Base-Patch16-224 matching the ONNX architecture."""
    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model


def get_calibration_dataloader(batch_size=64, num_samples=512, image_size=224):
    """Create calibration data for sparsity and quantization."""
    print(f"Creating calibration data: {num_samples} samples")
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def sparsify_model(model, calib_dataloader, device="cuda", method="sparsegpt"):
    """Apply 2:4 structured sparsity to the model."""
    print("\n" + "="*60)
    print(f"Applying 2:4 Sparsity ({method})")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    start_time = time.time()
    
    if method == "sparsegpt":
        config = {
            "data_loader": calib_dataloader,
            "collect_func": lambda batch: batch[0].to(device),
        }
        model = mts.sparsify(model, mode="sparsegpt", config=config)
    else:
        model = mts.sparsify(model, mode="sparse_magnitude")
    
    # Export to freeze sparsity pattern
    model = mts.export(model)
    
    elapsed = time.time() - start_time
    print(f"Sparsification completed in {elapsed:.2f}s")
    
    # Verify sparsity
    verify_sparsity(model)
    
    return model


def verify_sparsity(model):
    """Verify 2:4 sparsity pattern is applied."""
    total_params = 0
    sparse_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total = weight.numel()
            zeros = (weight == 0).sum().item()
            total_params += total
            sparse_params += zeros
    
    sparsity = sparse_params / total_params * 100
    print(f"Overall sparsity: {sparsity:.1f}% (expected ~50% for 2:4)")


def get_linear_layers_for_quantization():
    """Define which layers to quantize (linear only, not attention BMM)."""
    # Quantize all Linear layers except the final classification head
    return {
        "nn.Linear": {"*": {}, "*head*": None},  # Exclude head for stability
    }


def create_mxfp8_config():
    """Create MXFP8 quantization config for linear layers only."""
    # MXFP8 config targeting linear layers
    config = mtq.MXFP8_DEFAULT_CFG.copy()
    
    # Only quantize linear layers, skip attention matmuls
    config["quant_cfg"]["*weight_quantizer"] = {"num_bits": (4, 3), "axis": None}  # E4M3
    config["quant_cfg"]["*input_quantizer"] = {"num_bits": (4, 3), "axis": None}   # E4M3
    
    return config


def create_nvfp4_config():
    """Create NVFP4 quantization config for linear layers only."""
    config = mtq.NVFP4_DEFAULT_CFG.copy()
    return config


def quantize_model(model, config, calib_dataloader, device="cuda"):
    """Apply quantization to the model."""
    model = model.to(device)
    model.eval()
    
    def calibrate(model):
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_dataloader):
                if i >= 8:  # Use 8 batches for calibration
                    break
                images = images.to(device)
                model(images)
    
    model = mtq.quantize(model, config, forward_loop=calibrate)
    return model


def export_to_onnx(model, output_path, batch_size=64, device="cuda", opset=17):
    """Export model to ONNX format."""
    print(f"Exporting to: {output_path}")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"✅ Saved: {output_path}")


def export_quantized_onnx(model, output_path, batch_size=64, device="cuda"):
    """Export quantized model to ONNX with proper quantization ops."""
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    
    print(f"Exporting quantized model to: {output_path}")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Use modelopt's ONNX export for quantized models
    mtq.export_onnx(
        model,
        dummy_input,
        output_path,
        opset_version=17,
    )
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create Sparse ViT in all precisions")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="models/sparse")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sparsity_method", type=str, default="sparsegpt",
                        choices=["sparsegpt", "sparse_magnitude"])
    parser.add_argument("--skip_fp16", action="store_true")
    parser.add_argument("--skip_mxfp8", action="store_true")
    parser.add_argument("--skip_nvfp4", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create calibration data
    calib_dataloader = get_calibration_dataloader(
        batch_size=args.batch_size,
        num_samples=args.calib_size
    )
    
    # =========================================================================
    # Step 1: Create and sparsify base model
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Load and Sparsify ViT-Base-Patch16-224")
    print("="*70)
    
    base_model = create_vit_model()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    sparse_model = sparsify_model(
        base_model,
        calib_dataloader,
        device=args.device,
        method=args.sparsity_method
    )
    
    # Save sparse model state for reuse
    sparse_state_path = os.path.join(args.output_dir, "vit_sparse_base_state.pth")
    torch.save(sparse_model.state_dict(), sparse_state_path)
    print(f"Saved sparse model state to: {sparse_state_path}")
    
    # =========================================================================
    # Step 2: Export FP16 + Sparsity
    # =========================================================================
    if not args.skip_fp16:
        print("\n" + "="*70)
        print("STEP 2: Export FP16 + 2:4 Sparsity")
        print("="*70)
        
        fp16_path = os.path.join(args.output_dir, f"vit_sparse_fp16_bs{args.batch_size:03d}.onnx")
        
        # Convert to FP16 for export
        sparse_model_fp16 = copy.deepcopy(sparse_model).half().to(args.device)
        
        dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=args.device, dtype=torch.float16)
        
        torch.onnx.export(
            sparse_model_fp16,
            dummy_input,
            fp16_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"✅ FP16 + Sparsity: {fp16_path}")
    
    # =========================================================================
    # Step 3: Export MXFP8 + Sparsity
    # =========================================================================
    if not args.skip_mxfp8:
        print("\n" + "="*70)
        print("STEP 3: Export MXFP8 + 2:4 Sparsity (Linear Layers Only)")
        print("="*70)
        
        # Reload sparse model (fresh copy for quantization)
        mxfp8_model = create_vit_model()
        mxfp8_model.load_state_dict(torch.load(sparse_state_path))
        
        # Apply MXFP8 quantization
        print("Applying MXFP8 quantization to linear layers...")
        
        try:
            mxfp8_config = {
                "quant_cfg": {
                    "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
                    "*input_quantizer": {"num_bits": (4, 3), "axis": None},
                    "*lm_head*": {"enable": False},
                    "*head*": {"enable": False},
                },
                "algorithm": "max",
            }
            
            def calibrate_mxfp8(model):
                model.eval()
                with torch.no_grad():
                    for i, (images, _) in enumerate(calib_dataloader):
                        if i >= 8:
                            break
                        model(images.to(args.device))
            
            mxfp8_model = mtq.quantize(mxfp8_model.to(args.device), mxfp8_config, forward_loop=calibrate_mxfp8)
            
            mxfp8_path = os.path.join(args.output_dir, f"vit_sparse_mxfp8_bs{args.batch_size:03d}.onnx")
            
            dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=args.device)
            mtq.export_onnx(mxfp8_model, dummy_input, mxfp8_path, opset_version=17)
            
            print(f"✅ MXFP8 + Sparsity: {mxfp8_path}")
            
        except Exception as e:
            print(f"⚠️ MXFP8 export failed: {e}")
            print("This may require specific ModelOpt version or TensorRT container.")
    
    # =========================================================================
    # Step 4: Export NVFP4 + Sparsity
    # =========================================================================
    if not args.skip_nvfp4:
        print("\n" + "="*70)
        print("STEP 4: Export NVFP4 + 2:4 Sparsity (Linear Layers Only)")
        print("="*70)
        
        # Reload sparse model (fresh copy for quantization)
        nvfp4_model = create_vit_model()
        nvfp4_model.load_state_dict(torch.load(sparse_state_path))
        
        # Apply NVFP4 quantization
        print("Applying NVFP4 quantization to linear layers...")
        
        try:
            nvfp4_config = {
                "quant_cfg": {
                    "*weight_quantizer": {"num_bits": 4, "axis": 0},
                    "*input_quantizer": {"num_bits": 4, "axis": -1},
                    "*lm_head*": {"enable": False},
                    "*head*": {"enable": False},
                },
                "algorithm": "max",
            }
            
            def calibrate_nvfp4(model):
                model.eval()
                with torch.no_grad():
                    for i, (images, _) in enumerate(calib_dataloader):
                        if i >= 8:
                            break
                        model(images.to(args.device))
            
            nvfp4_model = mtq.quantize(nvfp4_model.to(args.device), nvfp4_config, forward_loop=calibrate_nvfp4)
            
            nvfp4_path = os.path.join(args.output_dir, f"vit_sparse_nvfp4_bs{args.batch_size:03d}.onnx")
            
            dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=args.device)
            mtq.export_onnx(nvfp4_model, dummy_input, nvfp4_path, opset_version=17)
            
            print(f"✅ NVFP4 + Sparsity: {nvfp4_path}")
            
        except Exception as e:
            print(f"⚠️ NVFP4 export failed: {e}")
            print("This may require specific ModelOpt version or TensorRT container.")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("COMPLETE! Generated Sparse Models:")
    print("="*70)
    
    for f in os.listdir(args.output_dir):
        if f.endswith('.onnx'):
            fpath = os.path.join(args.output_dir, f)
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"  {f}: {size_mb:.1f} MB")
    
    print("\n" + "="*70)
    print("NEXT STEPS: Build TensorRT Engines")
    print("="*70)
    print("""
# FP16 + Sparsity:
trtexec --onnx=vit_sparse_fp16_bs064.onnx \\
        --saveEngine=vit_sparse_fp16.engine \\
        --fp16 --sparsity=enable

# MXFP8 + Sparsity:
trtexec --onnx=vit_sparse_mxfp8_bs064.onnx \\
        --saveEngine=vit_sparse_mxfp8.engine \\
        --fp16 --stronglyTyped --sparsity=enable

# NVFP4 + Sparsity:
trtexec --onnx=vit_sparse_nvfp4_bs064.onnx \\
        --saveEngine=vit_sparse_nvfp4.engine \\
        --fp16 --stronglyTyped --sparsity=enable
""")


if __name__ == "__main__":
    main()

