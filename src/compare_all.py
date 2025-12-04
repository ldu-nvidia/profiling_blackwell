#!/usr/bin/env python3
"""
Compare All Profiling Results - FP16, MXFP8, NVFP4 with CUDA and TensorRT

Generates comprehensive comparison across all 6 configurations:
- FP16 + CUDA vs FP16 + TensorRT
- MXFP8 + CUDA vs MXFP8 + TensorRT  
- NVFP4 + CUDA vs NVFP4 + TensorRT
- Cross-model comparisons

Usage:
    python compare_all_results.py
    python compare_all_results.py --results-dir results/nsight-systems/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def find_latest_results(results_dir: str) -> Dict[str, Dict]:
    """Find the latest profiling results for each configuration"""
    results = {}
    
    # Expected configurations
    configs = [
        ('fp16', 'cuda'),
        ('fp16', 'trt'),
        ('mxfp8', 'cuda'),
        ('mxfp8', 'trt'),
        ('nvfp4', 'cuda'),
        ('nvfp4', 'trt'),
    ]
    
    results_path = Path(results_dir)
    
    for model, provider in configs:
        config_name = f"vit_{model}_{provider}"
        pattern = f"vit_{model}_{provider}_*"
        
        # Find all matching directories
        matching_dirs = sorted(results_path.glob(pattern), reverse=True)
        
        if matching_dirs:
            latest_dir = matching_dirs[0]
            metrics_file = latest_dir / "metrics.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        data['result_dir'] = str(latest_dir)
                        results[config_name] = data
                        print(f"  ✓ Found: {config_name} ({latest_dir.name})")
                except Exception as e:
                    print(f"  ✗ Error loading {config_name}: {e}")
            else:
                print(f"  ⚠ No metrics.json in {latest_dir}")
        else:
            print(f"  ⚠ Not found: {config_name}")
    
    return results


def calculate_speedup(baseline: float, optimized: float) -> Tuple[float, float]:
    """Calculate speedup factor and improvement percentage"""
    if optimized > 0:
        speedup = baseline / optimized
        improvement = ((baseline - optimized) / baseline) * 100
        return speedup, improvement
    return 0.0, 0.0


def generate_comparison_report(results: Dict[str, Dict]) -> str:
    """Generate comprehensive comparison report"""
    report = []
    
    report.append("=" * 80)
    report.append("COMPREHENSIVE PROFILING COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Overview table
    report.append("=" * 80)
    report.append("OVERVIEW - ALL CONFIGURATIONS")
    report.append("=" * 80)
    report.append("")
    
    header = f"{'Configuration':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15} {'Provider':<15}"
    report.append(header)
    report.append("-" * 80)
    
    for config_name in sorted(results.keys()):
        data = results[config_name]
        mean_lat = data.get('mean_latency_ms', 0)
        p95_lat = data.get('p95_latency_ms', 0)
        throughput = data.get('throughput_imgs_per_sec', 0)
        provider = data.get('provider_short', 'N/A')
        
        row = f"{config_name:<20} {mean_lat:<12.2f} {p95_lat:<12.2f} {throughput:<15.2f} {provider:<15}"
        report.append(row)
    
    report.append("")
    
    # TensorRT Speedup Analysis
    report.append("=" * 80)
    report.append("TENSORRT SPEEDUP ANALYSIS (TRT vs CUDA)")
    report.append("=" * 80)
    report.append("")
    
    models = ['fp16', 'mxfp8', 'nvfp4']
    
    for model in models:
        cuda_key = f"vit_{model}_cuda"
        trt_key = f"vit_{model}_trt"
        
        if cuda_key in results and trt_key in results:
            cuda_data = results[cuda_key]
            trt_data = results[trt_key]
            
            cuda_latency = cuda_data.get('mean_latency_ms', 0)
            trt_latency = trt_data.get('mean_latency_ms', 0)
            
            cuda_throughput = cuda_data.get('throughput_imgs_per_sec', 0)
            trt_throughput = trt_data.get('throughput_imgs_per_sec', 0)
            
            speedup, improvement = calculate_speedup(cuda_latency, trt_latency)
            throughput_gain = ((trt_throughput - cuda_throughput) / cuda_throughput * 100) if cuda_throughput > 0 else 0
            
            report.append(f"--- {model.upper()} Model ---")
            report.append(f"  CUDA Latency:      {cuda_latency:8.2f} ms")
            report.append(f"  TensorRT Latency:  {trt_latency:8.2f} ms")
            report.append(f"  Speedup:           {speedup:8.2f}x")
            report.append(f"  Improvement:       {improvement:8.2f}%")
            report.append(f"  Throughput Gain:   {throughput_gain:8.2f}%")
            report.append("")
    
    # Cross-Model Comparison (CUDA)
    report.append("=" * 80)
    report.append("CROSS-MODEL COMPARISON (Using CUDA Provider)")
    report.append("=" * 80)
    report.append("")
    
    cuda_configs = [f"vit_{m}_cuda" for m in models]
    available_cuda = [c for c in cuda_configs if c in results]
    
    if len(available_cuda) >= 2:
        baseline_key = available_cuda[0]
        baseline_data = results[baseline_key]
        baseline_latency = baseline_data.get('mean_latency_ms', 0)
        
        report.append(f"Baseline: {baseline_key} ({baseline_latency:.2f} ms)")
        report.append("")
        
        for config_key in available_cuda[1:]:
            data = results[config_key]
            latency = data.get('mean_latency_ms', 0)
            speedup, improvement = calculate_speedup(baseline_latency, latency)
            
            report.append(f"  vs {config_key}:")
            report.append(f"    Latency:     {latency:8.2f} ms")
            report.append(f"    Speedup:     {speedup:8.2f}x")
            report.append(f"    Improvement: {improvement:8.2f}%")
            report.append("")
    
    # Cross-Model Comparison (TensorRT)
    report.append("=" * 80)
    report.append("CROSS-MODEL COMPARISON (Using TensorRT Provider)")
    report.append("=" * 80)
    report.append("")
    
    trt_configs = [f"vit_{m}_trt" for m in models]
    available_trt = [c for c in trt_configs if c in results]
    
    if len(available_trt) >= 2:
        baseline_key = available_trt[0]
        baseline_data = results[baseline_key]
        baseline_latency = baseline_data.get('mean_latency_ms', 0)
        
        report.append(f"Baseline: {baseline_key} ({baseline_latency:.2f} ms)")
        report.append("")
        
        for config_key in available_trt[1:]:
            data = results[config_key]
            latency = data.get('mean_latency_ms', 0)
            speedup, improvement = calculate_speedup(baseline_latency, latency)
            
            report.append(f"  vs {config_key}:")
            report.append(f"    Latency:     {latency:8.2f} ms")
            report.append(f"    Speedup:     {speedup:8.2f}x")
            report.append(f"    Improvement: {improvement:8.2f}%")
            report.append("")
    
    # Model Size vs Performance
    report.append("=" * 80)
    report.append("MODEL SIZE vs PERFORMANCE")
    report.append("=" * 80)
    report.append("")
    
    model_sizes = {
        'fp16': 166,   # MB
        'mxfp8': 87,   # MB  
        'nvfp4': 48,   # MB
    }
    
    report.append(f"{'Model':<10} {'Size (MB)':<12} {'CUDA (ms)':<12} {'TRT (ms)':<12} {'Best Throughput':<15}")
    report.append("-" * 70)
    
    for model in models:
        cuda_key = f"vit_{model}_cuda"
        trt_key = f"vit_{model}_trt"
        
        size = model_sizes.get(model, 0)
        cuda_lat = results.get(cuda_key, {}).get('mean_latency_ms', 0)
        trt_lat = results.get(trt_key, {}).get('mean_latency_ms', 0)
        
        cuda_tput = results.get(cuda_key, {}).get('throughput_imgs_per_sec', 0)
        trt_tput = results.get(trt_key, {}).get('throughput_imgs_per_sec', 0)
        best_tput = max(cuda_tput, trt_tput)
        
        report.append(f"{model.upper():<10} {size:<12} {cuda_lat:<12.2f} {trt_lat:<12.2f} {best_tput:<15.2f}")
    
    report.append("")
    
    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Find best configuration
    best_config = None
    best_throughput = 0
    
    for config_name, data in results.items():
        throughput = data.get('throughput_imgs_per_sec', 0)
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = config_name
    
    if best_config:
        best_data = results[best_config]
        report.append(f"Best Configuration: {best_config}")
        report.append(f"  Throughput:  {best_throughput:.2f} images/sec")
        report.append(f"  Mean Latency: {best_data.get('mean_latency_ms', 0):.2f} ms")
        report.append(f"  P95 Latency:  {best_data.get('p95_latency_ms', 0):.2f} ms")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Compare all profiling results')
    parser.add_argument('--results-dir', type=str, default='results/nsight-systems/',
                       help='Directory containing profiling results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for comparison report')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LOADING PROFILING RESULTS")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print("")
    
    if not os.path.exists(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        print("Run ./run_all_profiles.sh first to generate results.")
        sys.exit(1)
    
    # Find and load results
    results = find_latest_results(args.results_dir)
    
    print("")
    print(f"Loaded {len(results)} configurations")
    print("")
    
    if not results:
        print("ERROR: No results found!")
        print("Run ./run_all_profiles.sh first to generate results.")
        sys.exit(1)
    
    # Generate report
    report = generate_comparison_report(results)
    
    # Output report
    print(report)
    
    # Save report
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.results_dir, f"COMPARISON_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print("")
    print(f"Report saved to: {output_path}")
    
    # Also save as JSON for further analysis
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()

