#!/usr/bin/env python3
"""
Compare Benchmark Results between FP16 and MXFP8 Models
"""

import json
import sys
import os
from typing import List, Dict
import glob

def load_results(pattern: str = "benchmark_results_*.json") -> List[Dict]:
    """Load all benchmark result files matching pattern"""
    files = glob.glob(pattern)
    results = []
    
    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results


def compare_models(results: List[Dict]):
    """Compare benchmark results"""
    if len(results) < 2:
        print(f"Need at least 2 results to compare, got {len(results)}")
        return
    
    print(f"\n{'='*80}")
    print(f"{'MODEL COMPARISON':^80}")
    print(f"{'='*80}\n")
    
    # Organize by model name
    fp16_results = [r for r in results if 'fp16' in r['model_name'].lower()]
    mxfp8_results = [r for r in results if 'mxfp8' in r['model_name'].lower() or 'fp8' in r['model_name'].lower()]
    
    if not fp16_results:
        print("No FP16 results found")
        return
    if not mxfp8_results:
        print("No MXFP8 results found")
        return
    
    # Use the most recent result for each
    fp16 = fp16_results[-1]
    mxfp8 = mxfp8_results[-1]
    
    # Model information
    print(f"{'Model Comparison':-^80}")
    print(f"{'FP16 Model:':<30} {fp16['model_name']}")
    print(f"{'MXFP8 Model:':<30} {mxfp8['model_name']}")
    print(f"{'FP16 Providers:':<30} {fp16['providers']}")
    print(f"{'MXFP8 Providers:':<30} {mxfp8['providers']}")
    
    # Compare latency
    print(f"\n{'Latency Comparison (ms)':-^80}")
    metrics = [
        ('Mean Latency', 'mean_latency_ms'),
        ('Median Latency', 'median_latency_ms'),
        ('Min Latency', 'min_latency_ms'),
        ('Max Latency', 'max_latency_ms'),
        ('P95 Latency', 'p95_latency_ms'),
        ('P99 Latency', 'p99_latency_ms'),
    ]
    
    print(f"{'Metric':<20} {'FP16':>12} {'MXFP8':>12} {'Speedup':>12} {'% Improvement':>15}")
    print(f"{'-'*80}")
    
    for metric_name, metric_key in metrics:
        fp16_val = fp16[metric_key]
        mxfp8_val = mxfp8[metric_key]
        speedup = fp16_val / mxfp8_val
        improvement = ((fp16_val - mxfp8_val) / fp16_val) * 100
        
        print(f"{metric_name:<20} {fp16_val:>12.3f} {mxfp8_val:>12.3f} {speedup:>12.3f}x {improvement:>14.2f}%")
    
    # Compare throughput
    print(f"\n{'Throughput Comparison':-^80}")
    fp16_throughput = fp16['throughput_imgs_per_sec']
    mxfp8_throughput = mxfp8['throughput_imgs_per_sec']
    throughput_gain = ((mxfp8_throughput - fp16_throughput) / fp16_throughput) * 100
    
    print(f"{'FP16 Throughput:':<30} {fp16_throughput:>12.2f} images/sec")
    print(f"{'MXFP8 Throughput:':<30} {mxfp8_throughput:>12.2f} images/sec")
    print(f"{'Throughput Gain:':<30} {throughput_gain:>12.2f}%")
    
    # Compare memory
    print(f"\n{'Memory Comparison':-^80}")
    fp16_mem = fp16['memory_after_mb']
    mxfp8_mem = mxfp8['memory_after_mb']
    mem_reduction = ((fp16_mem - mxfp8_mem) / fp16_mem) * 100
    
    print(f"{'FP16 Memory:':<30} {fp16_mem:>12.2f} MB")
    print(f"{'MXFP8 Memory:':<30} {mxfp8_mem:>12.2f} MB")
    print(f"{'Memory Reduction:':<30} {mem_reduction:>12.2f}%")
    
    # Model size comparison
    fp16_size = os.path.getsize(fp16['model_path']) / (1024 * 1024)  # MB
    mxfp8_size = os.path.getsize(mxfp8['model_path']) / (1024 * 1024)  # MB
    size_reduction = ((fp16_size - mxfp8_size) / fp16_size) * 100
    
    print(f"\n{'Model Size Comparison':-^80}")
    print(f"{'FP16 Model Size:':<30} {fp16_size:>12.2f} MB")
    print(f"{'MXFP8 Model Size:':<30} {mxfp8_size:>12.2f} MB")
    print(f"{'Size Reduction:':<30} {size_reduction:>12.2f}%")
    
    # Summary
    print(f"\n{'Summary':-^80}")
    print(f"✓ Latency improvement:    {improvement:>6.2f}% (MXFP8 is {speedup:.2f}x faster)")
    print(f"✓ Throughput gain:        {throughput_gain:>6.2f}%")
    print(f"✓ Memory reduction:       {mem_reduction:>6.2f}%")
    print(f"✓ Model size reduction:   {size_reduction:>6.2f}%")
    
    print(f"\n{'='*80}\n")
    
    # Save comparison
    comparison = {
        'fp16_model': fp16['model_name'],
        'mxfp8_model': mxfp8['model_name'],
        'latency_improvement_percent': improvement,
        'speedup_factor': speedup,
        'throughput_gain_percent': throughput_gain,
        'memory_reduction_percent': mem_reduction,
        'model_size_reduction_percent': size_reduction,
        'fp16_mean_latency_ms': fp16_val,
        'mxfp8_mean_latency_ms': mxfp8_val,
        'fp16_throughput': fp16_throughput,
        'mxfp8_throughput': mxfp8_throughput,
    }
    
    with open('comparison_summary.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("Comparison saved to: comparison_summary.json")


def main():
    if len(sys.argv) > 1:
        # Load specific files
        results = []
        for file in sys.argv[1:]:
            with open(file, 'r') as f:
                results.append(json.load(f))
    else:
        # Load all results in current directory
        results = load_results()
    
    if not results:
        print("No benchmark results found!")
        print("Usage: python compare_results.py [result1.json result2.json ...]")
        sys.exit(1)
    
    compare_models(results)


if __name__ == "__main__":
    main()

