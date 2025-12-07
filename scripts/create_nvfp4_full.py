#!/usr/bin/env python3
"""
Create Full NVFP4 Model (FP4 for BOTH linears AND attention)

This adds TRT_FP4DynamicQuantize to attention MatMuls (Q@K, A@V),
making the entire model use FP4 tensor cores.
"""

import onnx
from onnx import helper, TensorProto
import os
import argparse


def add_fp4_to_attention(input_path, output_path):
    print(f"Loading: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph
    
    # Find attention MatMuls (Q@K and A@V)
    attention_matmuls = []
    for node in graph.node:
        if node.op_type == 'MatMul':
            name_lower = node.name.lower()
            if '/attn/matmul' in name_lower and 'qkv' not in name_lower and 'proj' not in name_lower:
                attention_matmuls.append(node)
    
    print(f"Found {len(attention_matmuls)} attention MatMuls to quantize")
    
    # Find existing FP4 node as template
    fp4_template = None
    for node in graph.node:
        if node.op_type == 'TRT_FP4DynamicQuantize':
            fp4_template = node
            break
    
    if not fp4_template:
        print("ERROR: No TRT_FP4DynamicQuantize template found!")
        return
    
    print(f"Using template: {fp4_template.name}")
    print(f"  Attributes: {[(a.name, a.i) for a in fp4_template.attribute]}")
    
    nodes_to_add = []
    input_replacements = {}
    
    for idx, matmul in enumerate(attention_matmuls):
        for inp_idx, inp_name in enumerate(matmul.input):
            if inp_name in input_replacements:
                continue
            
            # Unique names
            base = f"attn_fp4_{idx}_{inp_idx}"
            q_name = f"/{base}/TRT_FP4DynamicQuantize"
            dq_name = f"/{base}/DequantizeLinear"
            scale_name = f"/{base}/scale"
            q_output = f"{base}_q_output"
            q_scale_output = f"{base}_scale_output"
            dq_output = f"{base}_dq_output"
            
            # Add scale initializer
            scale_tensor = helper.make_tensor(scale_name, TensorProto.FLOAT16, [1], [1.0])
            graph.initializer.append(scale_tensor)
            
            # TRT_FP4DynamicQuantize
            fp4_q = helper.make_node(
                "TRT_FP4DynamicQuantize",
                inputs=[inp_name, scale_name],
                outputs=[q_output, q_scale_output],
                name=q_name,
                domain="trt.plugins"
            )
            for attr in fp4_template.attribute:
                fp4_q.attribute.append(helper.make_attribute(attr.name, attr.i))
            
            # DequantizeLinear
            dq = helper.make_node(
                "DequantizeLinear",
                inputs=[q_output, q_scale_output],
                outputs=[dq_output],
                name=dq_name
            )
            
            nodes_to_add.extend([fp4_q, dq])
            input_replacements[inp_name] = dq_output
    
    # Update attention MatMul inputs
    for matmul in attention_matmuls:
        new_inputs = [input_replacements.get(inp, inp) for inp in matmul.input]
        matmul.ClearField('input')
        matmul.input.extend(new_inputs)
    
    # Add nodes
    graph.node.extend(nodes_to_add)
    
    # Save
    print(f"Saving: {output_path}")
    onnx.save(model, output_path)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")
    
    # Verify
    model_out = onnx.load(output_path)
    from collections import Counter
    ops = Counter(n.op_type for n in model_out.graph.node)
    
    print(f"\n=== Quantization Summary ===")
    print(f"TRT_FP4DynamicQuantize: {ops.get('TRT_FP4DynamicQuantize', 0)}")
    print(f"  - Linear layers: 49 (original)")
    print(f"  - Attention: {ops.get('TRT_FP4DynamicQuantize', 0) - 49} (new)")
    print(f"DequantizeLinear: {ops.get('DequantizeLinear', 0)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/workspace/models/vit_nvfp4_bs_064.onnx')
    parser.add_argument('--output', default='/workspace/models/vit_nvfp4_full.onnx')
    args = parser.parse_args()
    
    add_fp4_to_attention(args.input, args.output)

