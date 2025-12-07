#!/usr/bin/env python3
"""
Create NVFP4 + FP8 Attention Hybrid Model

This script takes the existing NVFP4 model and adds FP8 quantization
ONLY to the attention MatMuls (Q@K and A@V), which are not connected
to the FP4 weight quantization.

Configuration:
- Weights (Linear layers): NVFP4 (E2M1) - from existing model
- Attention Q@K, A@V: FP8 (E4M3) - added by this script
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import argparse
import os


def add_fp8_to_attention(input_path, output_path):
    """Add FP8 Q/DQ nodes around attention MatMuls only."""
    
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph
    
    # Build output->node map
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node
    
    # Find attention MatMuls (Q@K and A@V, not qkv/proj linears)
    attention_matmuls = []
    for node in graph.node:
        if node.op_type == 'MatMul':
            name_lower = node.name.lower()
            # Match /attn/MatMul and /attn/MatMul_1 but not qkv or proj
            if '/attn/matmul' in name_lower and 'qkv' not in name_lower and 'proj' not in name_lower:
                # Check inputs are NOT from FP4 quantization
                safe = True
                for inp in node.input:
                    producer = output_to_node.get(inp)
                    if producer and 'FP4' in producer.op_type:
                        safe = False
                        break
                if safe:
                    attention_matmuls.append(node)
    
    print(f"Found {len(attention_matmuls)} attention MatMuls to quantize")
    
    if len(attention_matmuls) == 0:
        print("ERROR: No attention MatMuls found!")
        return
    
    # Create FP8 scale initializer (shared)
    fp8_scale = helper.make_tensor(
        name="fp8_scale_shared",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0]  # Will be calibrated by TRT
    )
    graph.initializer.append(fp8_scale)
    
    nodes_to_add = []
    input_replacements = {}  # old_input -> new_input
    
    for idx, matmul in enumerate(attention_matmuls):
        print(f"  Adding FP8 to: {matmul.name}")
        
        for inp_idx, inp_name in enumerate(matmul.input):
            # Skip if already processed
            if inp_name in input_replacements:
                continue
            
            # Create unique names
            q_name = f"fp8_q_{idx}_{inp_idx}"
            dq_name = f"fp8_dq_{idx}_{inp_idx}"
            q_output = f"{inp_name}_fp8_q"
            dq_output = f"{inp_name}_fp8"
            
            # QuantizeLinear: input -> FP8
            q_node = helper.make_node(
                "QuantizeLinear",
                inputs=[inp_name, "fp8_scale_shared"],
                outputs=[q_output],
                name=q_name
            )
            # Set output type to FP8
            q_node.attribute.append(helper.make_attribute("output_dtype", TensorProto.FLOAT8E4M3FN))
            
            # DequantizeLinear: FP8 -> FP16
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[q_output, "fp8_scale_shared"],
                outputs=[dq_output],
                name=dq_name
            )
            
            nodes_to_add.extend([q_node, dq_node])
            input_replacements[inp_name] = dq_output
    
    # Replace inputs in attention MatMuls
    for matmul in attention_matmuls:
        new_inputs = []
        for inp in matmul.input:
            new_inputs.append(input_replacements.get(inp, inp))
        matmul.ClearField('input')
        matmul.input.extend(new_inputs)
    
    # Add new nodes to graph
    graph.node.extend(nodes_to_add)
    
    # Update opset to support FP8
    found_opset = False
    for opset in model.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            if opset.version < 19:
                opset.version = 19  # FP8 requires opset 19+
            found_opset = True
    
    if not found_opset:
        model.opset_import.append(helper.make_opsetid('', 19))
    
    # Save
    print(f"Saving to: {output_path}")
    onnx.save(model, output_path)
    
    # Verify
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")
    
    # Count nodes
    model_out = onnx.load(output_path)
    from collections import Counter
    ops = Counter(n.op_type for n in model_out.graph.node)
    
    print("\n=== Quantization Summary ===")
    print(f"  TRT_FP4DynamicQuantize: {ops.get('TRT_FP4DynamicQuantize', 0)} (weights)")
    print(f"  DequantizeLinear: {ops.get('DequantizeLinear', 0)} (includes FP4 + FP8)")
    print(f"  QuantizeLinear: {ops.get('QuantizeLinear', 0)} (FP8 attention)")
    
    print("\n✅ Hybrid NVFP4+FP8 model created!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/workspace/models/vit_nvfp4_bs_064.onnx')
    parser.add_argument('--output', default='/workspace/models/vit_nvfp4_fp8attn_v2.onnx')
    args = parser.parse_args()
    
    add_fp8_to_attention(args.input, args.output)


if __name__ == '__main__':
    main()

