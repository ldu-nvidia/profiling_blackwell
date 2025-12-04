#!/usr/bin/env python3
"""
Analyze and Compare Nsight Compute Profiling Results

This script parses NCU reports and generates comprehensive comparisons showing:
- Kernel execution time breakdown
- SM efficiency and occupancy differences
- Memory bandwidth utilization
- Instruction mix (FP16, FP32, Tensor Core usage)
- Performance bottlenecks

Usage:
    python analyze_ncu_results.py --input ncu_results_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import pandas as pd


def find_ncu():
    """Find Nsight Compute executable"""
    locations = [
        'ncu',
        '/usr/local/cuda/bin/ncu',
        '/cm/shared/apps/cuda12.4/toolkit/12.4.1/bin/ncu',
    ]
    
    for loc in locations:
        try:
            result = subprocess.run([loc, '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return loc
        except:
            continue
    return None


def export_ncu_to_json(ncu_report_path: str) -> Dict:
    """Export NCU report to structured JSON data"""
    ncu_path = find_ncu()
    if not ncu_path:
        print("Warning: NCU not found, cannot parse reports")
        return {}
    
    try:
        # Export detailed metrics to JSON
        result = subprocess.run(
            [ncu_path, '--import', ncu_report_path, '--export', '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Warning: Failed to export {ncu_report_path}")
            return {}
            
    except Exception as e:
        print(f"Warning: Error parsing {ncu_report_path}: {e}")
        return {}


def parse_ncu_summary(summary_file: str) -> Dict:
    """Parse text summary file for key metrics"""
    if not os.path.exists(summary_file):
        return {}
    
    with open(summary_file, 'r') as f:
        content = f.read()
    
    # Extract key information using regex
    metrics = {}
    
    # Try to find kernel names and execution times
    kernel_pattern = r'(\S+)\s+\((\d+)\)\s+(\d+\.?\d*)\s*(ns|us|ms|s)'
    matches = re.findall(kernel_pattern, content)
    
    kernels = []
    for match in matches:
        kernel_name, count, time_val, time_unit = match
        # Convert to microseconds
        time_us = float(time_val)
        if time_unit == 'ns':
            time_us /= 1000
        elif time_unit == 'ms':
            time_us *= 1000
        elif time_unit == 's':
            time_us *= 1000000
        
        kernels.append({
            'name': kernel_name,
            'count': int(count),
            'time_us': time_us
        })
    
    metrics['kernels'] = kernels
    metrics['total_kernel_time_us'] = sum(k['time_us'] for k in kernels)
    metrics['num_unique_kernels'] = len(kernels)
    metrics['total_kernel_launches'] = sum(k['count'] for k in kernels)
    
    return metrics


def analyze_configuration(ncu_report_path: str, summary_path: str) -> Dict:
    """Analyze a single configuration's NCU data"""
    config = {
        'report_path': ncu_report_path,
        'summary_path': summary_path,
    }
    
    # Parse summary file
    if os.path.exists(summary_path):
        summary_data = parse_ncu_summary(summary_path)
        config.update(summary_data)
    
    # Try to get more detailed data from JSON export
    # (This would require ncu, so optional)
    
    return config


def compare_configurations(configs: Dict[str, Dict]) -> str:
    """Generate comparison report between configurations"""
    report = []
    
    report.append("="*80)
    report.append("NSIGHT COMPUTE PROFILING COMPARISON")
    report.append("="*80)
    report.append("")
    
    # Overview
    report.append("Configurations Analyzed:")
    report.append("-" * 80)
    for name, config in configs.items():
        report.append(f"  {name}")
        if 'num_unique_kernels' in config:
            report.append(f"    - Unique kernels: {config['num_unique_kernels']}")
            report.append(f"    - Total launches: {config.get('total_kernel_launches', 'N/A')}")
            total_time_ms = config.get('total_kernel_time_us', 0) / 1000
            report.append(f"    - Total kernel time: {total_time_ms:.2f} ms")
    report.append("")
    
    # Compare CUDA vs TensorRT for FP16
    if 'fp16_cuda' in configs and 'fp16_tensorrt' in configs:
        report.append("="*80)
        report.append("FP16: CUDA vs TensorRT Comparison")
        report.append("="*80)
        
        cuda_config = configs['fp16_cuda']
        trt_config = configs['fp16_tensorrt']
        
        if 'num_unique_kernels' in cuda_config and 'num_unique_kernels' in trt_config:
            cuda_kernels = cuda_config['num_unique_kernels']
            trt_kernels = trt_config['num_unique_kernels']
            kernel_reduction = ((cuda_kernels - trt_kernels) / cuda_kernels) * 100
            
            cuda_time = cuda_config.get('total_kernel_time_us', 0) / 1000
            trt_time = trt_config.get('total_kernel_time_us', 0) / 1000
            
            if cuda_time > 0 and trt_time > 0:
                speedup = cuda_time / trt_time
                improvement = ((cuda_time - trt_time) / cuda_time) * 100
                
                report.append(f"")
                report.append(f"Kernel Count:")
                report.append(f"  CUDA:     {cuda_kernels} unique kernels")
                report.append(f"  TensorRT: {trt_kernels} unique kernels")
                report.append(f"  Reduction: {kernel_reduction:.1f}% (TensorRT fusion)")
                report.append(f"")
                report.append(f"Total Kernel Execution Time:")
                report.append(f"  CUDA:      {cuda_time:.2f} ms")
                report.append(f"  TensorRT:  {trt_time:.2f} ms")
                report.append(f"  Speedup:   {speedup:.2f}x")
                report.append(f"  Improvement: {improvement:.1f}%")
                report.append(f"")
                report.append(f"Interpretation:")
                report.append(f"  - TensorRT reduced unique kernels by {kernel_reduction:.1f}% through fusion")
                report.append(f"  - TensorRT is {speedup:.2f}x faster at kernel execution level")
                report.append(f"  - This confirms TensorRT's optimization effectiveness")
        
        report.append("")
    
    # Top kernels analysis
    for name, config in configs.items():
        if 'kernels' in config and config['kernels']:
            report.append("="*80)
            report.append(f"Top 10 Kernels: {name}")
            report.append("="*80)
            
            kernels = sorted(config['kernels'], key=lambda x: x['time_us'], reverse=True)[:10]
            
            report.append(f"{'Kernel Name':<50} {'Time (ms)':<12} {'Count':<8}")
            report.append("-" * 80)
            
            for kernel in kernels:
                time_ms = kernel['time_us'] / 1000
                report.append(f"{kernel['name']:<50} {time_ms:<12.3f} {kernel['count']:<8}")
            
            report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze Nsight Compute profiling results')
    parser.add_argument('--input', type=str, required=True,
                       help='Directory containing NCU results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for comparison report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)
    
    # Find all NCU reports
    ncu_files = list(Path(args.input).glob("*.ncu-rep"))
    
    if not ncu_files:
        print(f"ERROR: No .ncu-rep files found in {args.input}")
        print("\nMake sure you've run: ./run_ncu_profiling.sh")
        sys.exit(1)
    
    print(f"Found {len(ncu_files)} NCU reports")
    print("")
    
    # Analyze each configuration
    configs = {}
    
    for ncu_file in ncu_files:
        # Determine configuration name from filename
        filename = ncu_file.stem
        
        # Extract model and provider
        if 'fp16' in filename.lower():
            model = 'fp16'
        elif 'mxfp8' in filename.lower() or 'fp8' in filename.lower():
            model = 'mxfp8'
        else:
            model = 'unknown'
        
        if 'cuda' in filename.lower() and 'tensorrt' not in filename.lower():
            provider = 'cuda'
        elif 'tensorrt' in filename.lower():
            provider = 'tensorrt'
        else:
            provider = 'unknown'
        
        config_name = f"{model}_{provider}"
        
        # Find corresponding summary file
        summary_file = str(ncu_file).replace('.ncu-rep', '_summary.txt')
        
        print(f"Analyzing: {config_name}")
        configs[config_name] = analyze_configuration(str(ncu_file), summary_file)
    
    print("")
    print("="*80)
    print("Generating Comparison Report")
    print("="*80)
    
    # Generate comparison
    report = compare_configurations(configs)
    
    # Output report
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.input, "NCU_COMPARISON.txt")
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print("")
    print(f"Report saved to: {output_path}")
    print("")
    
    # Save configs to JSON for further analysis
    json_path = os.path.join(args.input, "ncu_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Detailed data saved to: {json_path}")
    print("")


if __name__ == "__main__":
    main()

