#!/usr/bin/env python3
"""
Profile ONNX Model with Nsight Compute for Detailed Kernel Analysis

This script runs inference with Nsight Compute to capture detailed per-kernel metrics:
- SM efficiency and occupancy
- Memory throughput and bandwidth utilization
- Warp execution efficiency
- Instruction mix (FP16, FP32, INT, Tensor Core usage)
- Cache hit rates
- Roofline analysis

Usage:
    python profile_with_ncu.py --model vit_fp16_bs_064.onnx --provider cuda --output results/
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import onnxruntime as ort


def find_ncu():
    """Find Nsight Compute executable"""
    # Check common locations
    locations = [
        'ncu',  # In PATH
        '/usr/local/cuda/bin/ncu',
        '/usr/local/cuda-12/bin/ncu',
        '/usr/local/cuda-12.4/bin/ncu',
        '/cm/shared/apps/cuda12.4/toolkit/12.4.1/bin/ncu',
    ]
    
    for loc in locations:
        try:
            result = subprocess.run(
                [loc, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"Found Nsight Compute: {loc}")
                return loc
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return None


def run_inference_with_ncu(
    model_path: str,
    provider: str,
    output_dir: str,
    num_iterations: int = 10,
    ncu_metrics: str = "full"
):
    """
    Run ONNX model inference with Nsight Compute profiling
    
    Args:
        model_path: Path to ONNX model
        provider: 'cuda' or 'tensorrt'
        output_dir: Directory to save profiling results
        num_iterations: Number of inference iterations to profile
        ncu_metrics: Metric set ('full', 'basic', or custom)
    
    Returns:
        Path to NCU report file
    """
    ncu_path = find_ncu()
    if not ncu_path:
        print("ERROR: Nsight Compute (ncu) not found!")
        print("\nInstallation options:")
        print("  1. Load CUDA module: module load cuda12.4/toolkit/12.4.1")
        print("  2. Install CUDA Toolkit (includes ncu)")
        print("  3. Download standalone: https://developer.nvidia.com/nsight-compute")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_base = f"{model_name}_{provider}_ncu_{timestamp}"
    report_path = os.path.join(output_dir, report_base)
    
    # Create a simple inference script that ncu will profile
    inference_script = os.path.join(output_dir, f"_inference_{model_name}_{provider}.py")
    
    with open(inference_script, 'w') as f:
        f.write(f'''
import numpy as np
import onnxruntime as ort

# Setup
model_path = "{model_path}"
provider_map = {{
    'cuda': ['CUDAExecutionProvider'],
    'tensorrt': ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
}}
providers = provider_map["{provider}"]

# Create session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

# Get input info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

# Create input
if 'float16' in input_type:
    dummy_input = np.random.randn(*input_shape).astype(np.float16)
else:
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

output_name = session.get_outputs()[0].name

print(f"Model: {{model_path}}")
print(f"Provider: {{providers}}")
print(f"Input shape: {{input_shape}}")

# Warmup
print("Warming up...")
for i in range(5):
    _ = session.run([output_name], {{input_name: dummy_input}})

# Profile iterations
print(f"Running {{num_iterations}} profiled iterations...")
for i in range({num_iterations}):
    output = session.run([output_name], {{input_name: dummy_input}})
    if (i + 1) % 2 == 0:
        print(f"  Iteration {{i+1}}/{num_iterations}")

print("Profiling complete!")
''')
    
    # Build ncu command
    ncu_cmd = [
        ncu_path,
        '--set', ncu_metrics,  # Metric set: full, basic, or custom
        '--target-processes', 'all',  # Profile all GPU processes
        '--kernel-name-base', 'demangled',  # Use readable kernel names
        '--launch-skip', '5',  # Skip warmup iterations
        '--launch-count', str(num_iterations),  # Number of launches to profile
        '--export', report_path,  # Export report
        '--force-overwrite',  # Overwrite existing
        'python', inference_script
    ]
    
    # Additional useful flags for detailed analysis
    if ncu_metrics == "full":
        ncu_cmd.extend([
            '--metrics', 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum',  # FP ops
            '--metrics', 'dram__bytes.sum,lts__t_bytes.sum',  # Memory
            '--metrics', 'sm__warps_active.avg.pct_of_peak_sustained_active',  # Occupancy
        ])
    
    print("\n" + "="*80)
    print("STARTING NSIGHT COMPUTE PROFILING")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Provider: {provider}")
    print(f"Metric set: {ncu_metrics}")
    print(f"Iterations: {num_iterations}")
    print(f"Output: {report_path}.ncu-rep")
    print("\nThis will take several minutes (ncu replays each kernel ~30-50 times)...")
    print("="*80 + "\n")
    
    # Run ncu
    start_time = time.time()
    try:
        result = subprocess.run(ncu_cmd, check=True)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Profiling completed in {elapsed:.1f} seconds")
        print(f"✓ Report saved: {report_path}.ncu-rep")
        
        # Clean up temporary inference script
        os.remove(inference_script)
        
        return f"{report_path}.ncu-rep"
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Profiling failed with error code {e.returncode}")
        print("\nCommon issues:")
        print("  - Not on a GPU node (run: srun --gres=gpu:1 ... --pty bash)")
        print("  - CUDA driver version mismatch")
        print("  - Insufficient permissions")
        sys.exit(1)


def export_ncu_report_to_csv(ncu_report_path: str):
    """Export NCU report to CSV for analysis"""
    ncu_path = find_ncu()
    if not ncu_path:
        return None
    
    csv_path = ncu_report_path.replace('.ncu-rep', '_summary.csv')
    
    try:
        # Export summary statistics
        subprocess.run([
            ncu_path,
            '--import', ncu_report_path,
            '--csv',
            '--page', 'details',
            '--print-summary', 'per-kernel',
        ], stdout=open(csv_path, 'w'), check=True)
        
        print(f"✓ CSV export: {csv_path}")
        return csv_path
        
    except subprocess.CalledProcessError:
        print("⚠ CSV export failed (this is OK - report is still available)")
        return None


def main():
    parser = argparse.ArgumentParser(description='Profile ONNX model with Nsight Compute')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--provider', type=str, required=True, 
                       choices=['cuda', 'tensorrt'],
                       help='Execution provider')
    parser.add_argument('--output', type=str, default='ncu_results',
                       help='Output directory for reports')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations to profile')
    parser.add_argument('--metrics', type=str, default='full',
                       choices=['full', 'basic'],
                       help='Metric collection set')
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    # Run profiling
    report_path = run_inference_with_ncu(
        model_path=args.model,
        provider=args.provider,
        output_dir=args.output,
        num_iterations=args.iterations,
        ncu_metrics=args.metrics
    )
    
    # Export to CSV
    csv_path = export_ncu_report_to_csv(report_path)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - NCU Report: {report_path}")
    if csv_path:
        print(f"  - CSV Export: {csv_path}")
    
    print("\nView report:")
    print(f"  1. Download {report_path} to local machine")
    print(f"  2. Open with: nsight-compute-gui {Path(report_path).name}")
    
    print("\nGenerate text summary:")
    print(f"  ncu --import {report_path} --page details --print-summary per-kernel")
    print()


if __name__ == "__main__":
    main()

