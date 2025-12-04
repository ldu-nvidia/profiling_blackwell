#!/usr/bin/env python3
"""
Comprehensive ONNX Model Benchmarking with ONNX Runtime
Benchmarks Vision Transformer models with different precisions
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import psutil
from datetime import datetime
from typing import Dict, List, Tuple
import onnxruntime as ort

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ONNXBenchmark:
    """Benchmark ONNX models with various metrics"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize benchmark for an ONNX model
        
        Args:
            model_path: Path to the ONNX model
            providers: List of execution providers (e.g., ['CUDAExecutionProvider'])
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # Set providers
        if providers is None:
            # Try GPU first, fall back to CPU
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self.providers = ['CUDAExecutionProvider']
            elif 'TensorrtExecutionProvider' in available_providers:
                self.providers = ['TensorrtExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers
        
        print(f"Using providers: {self.providers}")
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model: {self.model_name}")
        print(f"Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"Output: {self.output_name}")
        
    def warmup(self, num_iterations: int = 10):
        """Warmup the model"""
        print(f"\nWarming up ({num_iterations} iterations)...")
        
        # Get input dtype from model
        input_type = self.session.get_inputs()[0].type
        if 'float16' in input_type:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float16)
        else:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        for i in range(num_iterations):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            if (i + 1) % 5 == 0:
                print(f"  Warmup {i + 1}/{num_iterations}")
        
        print("Warmup complete!")
    
    def benchmark_inference(
        self,
        num_iterations: int = 100,
        batch_size: int = None
    ) -> Dict:
        """
        Run inference benchmark
        
        Args:
            num_iterations: Number of iterations to run
            batch_size: Override batch size (None = use model's default)
        
        Returns:
            Dictionary with benchmark results
        """
        # Prepare input
        input_shape = list(self.input_shape)
        if batch_size is not None:
            input_shape[0] = batch_size
        
        # Get input dtype from model
        input_type = self.session.get_inputs()[0].type
        if 'float16' in input_type:
            input_dtype = np.float16
        else:
            input_dtype = np.float32
        
        print(f"\nBenchmarking with shape: {input_shape}, dtype: {input_dtype}")
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)
        
        # Benchmark
        latencies = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print(f"Running {num_iterations} iterations...")
        for i in range(num_iterations):
            start = time.perf_counter()
            outputs = self.session.run([self.output_name], {self.input_name: dummy_input})
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i + 1}/{num_iterations} - Latest: {latency_ms:.2f}ms")
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'providers': self.providers,
            'input_shape': input_shape,
            'num_iterations': num_iterations,
            'timestamp': datetime.now().isoformat(),
            
            # Latency statistics (ms)
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            
            # Throughput (images/sec)
            'throughput_imgs_per_sec': float(input_shape[0] / (np.mean(latencies) / 1000)),
            
            # Memory
            'memory_increase_mb': float(memory_after - memory_before),
            'memory_before_mb': float(memory_before),
            'memory_after_mb': float(memory_after),
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS: {results['model_name']}")
        print(f"{'='*70}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Providers: {results['providers']}")
        print(f"Input Shape: {results['input_shape']}")
        print(f"Iterations: {results['num_iterations']}")
        
        print(f"\n{' Latency (ms) ':-^70}")
        print(f"  Mean:    {results['mean_latency_ms']:8.3f} ms")
        print(f"  Median:  {results['median_latency_ms']:8.3f} ms")
        print(f"  Min:     {results['min_latency_ms']:8.3f} ms")
        print(f"  Max:     {results['max_latency_ms']:8.3f} ms")
        print(f"  Std Dev: {results['std_latency_ms']:8.3f} ms")
        print(f"  P50:     {results['p50_latency_ms']:8.3f} ms")
        print(f"  P95:     {results['p95_latency_ms']:8.3f} ms")
        print(f"  P99:     {results['p99_latency_ms']:8.3f} ms")
        
        print(f"\n{' Throughput ':-^70}")
        print(f"  {results['throughput_imgs_per_sec']:.2f} images/sec")
        
        print(f"\n{' Memory ':-^70}")
        print(f"  Before:   {results['memory_before_mb']:.2f} MB")
        print(f"  After:    {results['memory_after_mb']:.2f} MB")
        print(f"  Increase: {results['memory_increase_mb']:.2f} MB")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ONNX models')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    parser.add_argument('--provider', type=str, default='cuda', 
                       choices=['cuda', 'tensorrt', 'cpu'],
                       help='Execution provider')
    
    args = parser.parse_args()
    
    # Map provider argument to ONNX Runtime provider
    provider_map = {
        'cuda': ['CUDAExecutionProvider'],
        'tensorrt': ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
        'cpu': ['CPUExecutionProvider']
    }
    
    providers = provider_map[args.provider]
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"ONNX RUNTIME BENCHMARK")
    print(f"{'='*70}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"{'='*70}\n")
    
    # Create benchmark
    benchmark = ONNXBenchmark(args.model, providers=providers)
    
    # Warmup
    benchmark.warmup(num_iterations=args.warmup)
    
    # Run benchmark
    results = benchmark.benchmark_inference(
        num_iterations=args.iterations,
        batch_size=args.batch_size
    )
    
    # Print results
    benchmark.print_results(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        model_basename = os.path.splitext(os.path.basename(args.model))[0]
        output_path = f"benchmark_results_{model_basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

