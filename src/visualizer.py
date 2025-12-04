#!/usr/bin/env python3
"""
Visualize Profiling Results - Generate Charts and Plots

Creates comprehensive visualizations comparing:
- Latency across configurations
- Throughput comparison
- Kernel execution time breakdown
- Provider performance comparison (CUDA vs TensorRT)
- Model precision comparison (FP16 vs MXFP8)

Usage:
    python visualize_profiling_results.py --results results_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_benchmark_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON benchmark results from directory"""
    results = {}
    
    json_files = list(Path(results_dir).glob("*.json"))
    
    for json_file in json_files:
        filename = json_file.stem
        
        # Skip summary/comparison files
        if 'summary' in filename.lower() or 'comparison' in filename.lower():
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[filename] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def extract_config_info(filename: str) -> Dict[str, str]:
    """Extract model and provider from filename"""
    info = {}
    
    # Model
    if 'fp16' in filename.lower():
        info['model'] = 'FP16'
    elif 'mxfp8' in filename.lower() or 'fp8' in filename.lower():
        info['model'] = 'MXFP8'
    else:
        info['model'] = 'Unknown'
    
    # Provider
    if 'tensorrt' in filename.lower():
        info['provider'] = 'TensorRT'
    elif 'cuda' in filename.lower():
        info['provider'] = 'CUDA'
    elif 'cpu' in filename.lower():
        info['provider'] = 'CPU'
    else:
        info['provider'] = 'Unknown'
    
    info['config'] = f"{info['model']}\n{info['provider']}"
    
    return info


def plot_latency_comparison(results: Dict[str, Dict], output_dir: str):
    """Plot latency comparison across configurations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = []
    mean_latencies = []
    p95_latencies = []
    colors = []
    
    for name, data in results.items():
        config_info = extract_config_info(name)
        configs.append(config_info['config'])
        mean_latencies.append(data.get('mean_latency_ms', 0))
        p95_latencies.append(data.get('p95_latency_ms', 0))
        
        # Color by provider
        if 'TensorRT' in config_info['provider']:
            colors.append('#2ecc71')  # Green
        elif 'CUDA' in config_info['provider']:
            colors.append('#3498db')  # Blue
        else:
            colors.append('#95a5a6')  # Gray
    
    # Mean latency
    ax1.bar(configs, mean_latencies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('Mean Latency Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(mean_latencies):
        ax1.text(i, v + max(mean_latencies)*0.02, f'{v:.2f}ms', 
                ha='center', va='bottom', fontsize=9)
    
    # P95 latency
    ax2.bar(configs, p95_latencies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('P95 Latency (ms)', fontsize=12)
    ax2.set_title('P95 Latency Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(p95_latencies):
        ax2.text(i, v + max(p95_latencies)*0.02, f'{v:.2f}ms', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: latency_comparison.png")
    plt.close()


def plot_throughput_comparison(results: Dict[str, Dict], output_dir: str):
    """Plot throughput comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = []
    throughputs = []
    colors = []
    
    for name, data in results.items():
        config_info = extract_config_info(name)
        configs.append(config_info['config'])
        throughputs.append(data.get('throughput_imgs_per_sec', 0))
        
        if 'TensorRT' in config_info['provider']:
            colors.append('#2ecc71')
        elif 'CUDA' in config_info['provider']:
            colors.append('#3498db')
        else:
            colors.append('#95a5a6')
    
    ax.bar(configs, throughputs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Throughput (images/sec)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(throughputs):
        ax.text(i, v + max(throughputs)*0.02, f'{v:.0f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: throughput_comparison.png")
    plt.close()


def plot_provider_speedup(results: Dict[str, Dict], output_dir: str):
    """Plot TensorRT speedup over CUDA"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model
    models = {}
    for name, data in results.items():
        config_info = extract_config_info(name)
        model = config_info['model']
        provider = config_info['provider']
        
        if model not in models:
            models[model] = {}
        models[model][provider] = data.get('mean_latency_ms', 0)
    
    # Calculate speedups
    model_names = []
    speedups = []
    
    for model, providers in models.items():
        if 'CUDA' in providers and 'TensorRT' in providers:
            cuda_latency = providers['CUDA']
            trt_latency = providers['TensorRT']
            
            if trt_latency > 0:
                speedup = cuda_latency / trt_latency
                model_names.append(model)
                speedups.append(speedup)
    
    if speedups:
        bars = ax.bar(model_names, speedups, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)')
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('TensorRT Speedup vs CUDA', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(speedups):
            ax.text(i, v + 0.05, f'{v:.2f}x', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tensorrt_speedup.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: tensorrt_speedup.png")
    else:
        print("⚠ Cannot generate speedup plot - missing CUDA or TensorRT data")
    
    plt.close()


def plot_latency_distribution(results: Dict[str, Dict], output_dir: str):
    """Plot latency percentile comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results))
    width = 0.15
    
    percentiles = ['p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 
                   'min_latency_ms', 'max_latency_ms']
    labels = ['P50', 'P95', 'P99', 'Min', 'Max']
    colors_list = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    configs = []
    for name in results.keys():
        config_info = extract_config_info(name)
        configs.append(config_info['config'])
    
    for i, (percentile, label, color) in enumerate(zip(percentiles, labels, colors_list)):
        values = [results[name].get(percentile, 0) for name in results.keys()]
        ax.bar(x + i*width, values, width, label=label, color=color, alpha=0.7)
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(configs, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: latency_distribution.png")
    plt.close()


def create_summary_table(results: Dict[str, Dict], output_dir: str):
    """Create summary table as image"""
    data_rows = []
    
    for name, data in results.items():
        config_info = extract_config_info(name)
        
        row = {
            'Configuration': f"{config_info['model']} + {config_info['provider']}",
            'Mean Latency (ms)': f"{data.get('mean_latency_ms', 0):.2f}",
            'P95 Latency (ms)': f"{data.get('p95_latency_ms', 0):.2f}",
            'Throughput (img/s)': f"{data.get('throughput_imgs_per_sec', 0):.0f}",
            'Memory (MB)': f"{data.get('memory_after_mb', 0):.0f}",
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, len(data_rows) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#3498db']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(data_rows) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: summary_table.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize profiling results')
    parser.add_argument('--results', type=str, required=True,
                       help='Directory containing benchmark JSON results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"ERROR: Results directory not found: {args.results}")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from: {args.results}")
    results = load_benchmark_results(args.results)
    
    if not results:
        print("ERROR: No valid benchmark results found!")
        print("Make sure directory contains JSON files from benchmark_onnx.py")
        sys.exit(1)
    
    print(f"Found {len(results)} configurations")
    print("")
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.results, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    print("")
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_latency_comparison(results, output_dir)
    plot_throughput_comparison(results, output_dir)
    plot_provider_speedup(results, output_dir)
    plot_latency_distribution(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("")
    print("="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated {5} visualization files in: {output_dir}")
    print("\nFiles created:")
    print("  - latency_comparison.png")
    print("  - throughput_comparison.png")
    print("  - tensorrt_speedup.png")
    print("  - latency_distribution.png")
    print("  - summary_table.png")
    print("")


if __name__ == "__main__":
    main()

