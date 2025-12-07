#!/usr/bin/env python3
"""
Create Full NVFP4 Model with FP4 for BOTH linears AND attention.

Uses static padding for known ViT dimensions:
- Sequence length 197 -> pad to 208 (divisible by 16)
- Head dim 64 -> already divisible by 16
"""

import onnx
from onnx import helper, TensorProto
import os
import argparse


def add_fp4_to_attention(input_path, output_path):
    print(f"Loading: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph
    
    # Find attention MatMuls
    attention_matmuls = []
    for node in graph.node:
        if node.op_type == 'MatMul':
            name_lower = node.name.lower()
            if '/attn/matmul' in name_lower and 'qkv' not in name_lower and 'proj' not in name_lower:
                attention_matmuls.append(node)
    
    print(f"Found {len(attention_matmuls)} attention MatMuls")
    
    # Get FP4 template attributes
    fp4_attrs = []
    for node in graph.node:
        if node.op_type == 'TRT_FP4DynamicQuantize':
            fp4_attrs = [(a.name, a.i) for a in node.attribute]
            break
    
    nodes_to_add = []
    input_replacements = {}
    
    # Known shapes for ViT:
    # Q: [64, 12, 197, 64] - last dim 64 % 16 = 0 (no pad needed)
    # K^T: [64, 12, 64, 197] - last dim 197 % 16 = 5, need pad 11
    # Softmax: [64, 12, 197, 197] - last dim 197 % 16 = 5, need pad 11
    # V: [64, 12, 197, 64] - last dim 64 % 16 = 0 (no pad needed)
    
    for idx, matmul in enumerate(attention_matmuls):
        for inp_idx, inp_name in enumerate(matmul.input):
            if inp_name in input_replacements:
                continue
            
            base = f"attn_fp4_{idx}_{inp_idx}"
            
            # Step 1: Get original shape
            orig_shape = f"{base}_orig_shape"
            nodes_to_add.append(helper.make_node(
                "Shape", [inp_name], [orig_shape], name=f"/{base}/Shape"
            ))
            
            # Step 2: Pad if last dim is 197 (static pad of 11)
            # Pads format for 4D: [d0_begin, d1_begin, d2_begin, d3_begin, d0_end, d1_end, d2_end, d3_end]
            # Pad only last dim at end: [0,0,0,0, 0,0,0,11]
            pads_197 = helper.make_tensor(f"{base}_pads197", TensorProto.INT64, [8], [0,0,0,0,0,0,0,11])
            pads_0 = helper.make_tensor(f"{base}_pads0", TensorProto.INT64, [8], [0,0,0,0,0,0,0,0])
            graph.initializer.extend([pads_197, pads_0])
            
            # We need to check last dim and conditionally pad
            # For simplicity, let's just pad and then the extra 0s won't hurt
            # Actually TRT will complain if we pad unnecessarily
            
            # Let's check which inputs need padding by their position in the matmul
            # For Q@K^T (MatMul): input[0]=Q [B,H,S,D], input[1]=K^T [B,H,D,S]
            # For A@V (MatMul_1): input[0]=A [B,H,S,S], input[1]=V [B,H,S,D]
            
            # K^T and A have 197 as last dim, need padding
            # Q and V have 64 as last dim, no padding needed
            
            # Determine if this input needs padding based on typical shapes
            # We'll use a trick: check if it's input[1] of Q@K (K^T) or input[0] of A@V (Softmax output)
            needs_pad = False
            if 'MatMul_1' not in matmul.name:  # Q@K matmul
                if inp_idx == 1:  # K^T needs padding
                    needs_pad = True
            else:  # A@V matmul
                if inp_idx == 0:  # Softmax output needs padding
                    needs_pad = True
            
            if needs_pad:
                print(f"  {base}: needs padding (197 -> 208)")
                pads_name = f"{base}_pads197"
                # Pad
                padded = f"{base}_padded"
                nodes_to_add.append(helper.make_node(
                    "Pad", [inp_name, pads_name], [padded],
                    name=f"/{base}/Pad", mode="constant"
                ))
                current_tensor = padded
                
                # Get padded shape
                padded_shape = f"{base}_padded_shape"
                nodes_to_add.append(helper.make_node(
                    "Shape", [padded], [padded_shape], name=f"/{base}/PaddedShape"
                ))
                shape_for_3d = padded_shape
            else:
                print(f"  {base}: no padding needed")
                current_tensor = inp_name
                shape_for_3d = orig_shape
            
            # Step 3: Reshape to 3D
            # Compute [B*H, X, Y]
            slice_s = helper.make_tensor(f"{base}_ss", TensorProto.INT64, [1], [0])
            slice_e2 = helper.make_tensor(f"{base}_se2", TensorProto.INT64, [1], [2])
            slice_e4 = helper.make_tensor(f"{base}_se4", TensorProto.INT64, [1], [4])
            slice_ax = helper.make_tensor(f"{base}_sax", TensorProto.INT64, [1], [0])
            graph.initializer.extend([slice_s, slice_e2, slice_e4, slice_ax])
            
            bh_dims = f"{base}_bh"
            nodes_to_add.append(helper.make_node(
                "Slice", [shape_for_3d, f"{base}_ss", f"{base}_se2", f"{base}_sax"],
                [bh_dims], name=f"/{base}/SliceBH"
            ))
            
            bh_prod = f"{base}_bhp"
            nodes_to_add.append(helper.make_node(
                "ReduceProd", [bh_dims], [bh_prod], name=f"/{base}/ProdBH", keepdims=1
            ))
            
            xy_dims = f"{base}_xy"
            nodes_to_add.append(helper.make_node(
                "Slice", [shape_for_3d, f"{base}_se2", f"{base}_se4", f"{base}_sax"],
                [xy_dims], name=f"/{base}/SliceXY"
            ))
            
            shape_3d = f"{base}_s3d"
            nodes_to_add.append(helper.make_node(
                "Concat", [bh_prod, xy_dims], [shape_3d], name=f"/{base}/Shape3D", axis=0
            ))
            
            tensor_3d = f"{base}_t3d"
            nodes_to_add.append(helper.make_node(
                "Reshape", [current_tensor, shape_3d], [tensor_3d], name=f"/{base}/To3D"
            ))
            
            # Step 4: FP4 Quantize
            const_scale = helper.make_tensor(f"{base}_cs", TensorProto.FLOAT16, [1], [1.0])
            const_dq_scale = helper.make_tensor(f"{base}_cds", TensorProto.FLOAT, [1], [1.0])
            graph.initializer.extend([const_scale, const_dq_scale])
            
            quant_out = f"{base}_q"
            scale_out = f"{base}_qs"
            fp4_node = helper.make_node(
                "TRT_FP4DynamicQuantize",
                [tensor_3d, f"{base}_cs"],
                [quant_out, scale_out],
                name=f"/{base}/FP4Q",
                domain="trt.plugins"
            )
            for name, val in fp4_attrs:
                fp4_node.attribute.append(helper.make_attribute(name, val))
            nodes_to_add.append(fp4_node)
            
            # Step 5: Dequantize
            dq_scale = f"{base}_dqs"
            nodes_to_add.append(helper.make_node(
                "DequantizeLinear", [scale_out, f"{base}_cds"],
                [dq_scale], name=f"/{base}/DQ1"
            ))
            
            dq_tensor_3d = f"{base}_dqt3d"
            nodes_to_add.append(helper.make_node(
                "DequantizeLinear", [quant_out, dq_scale],
                [dq_tensor_3d], name=f"/{base}/DQ2"
            ))
            
            # Step 6: Reshape back to 4D
            dq_tensor_4d = f"{base}_dqt4d"
            nodes_to_add.append(helper.make_node(
                "Reshape", [dq_tensor_3d, shape_for_3d], [dq_tensor_4d],
                name=f"/{base}/To4D"
            ))
            
            # Step 7: Slice to remove padding if we added it
            if needs_pad:
                slice_starts = helper.make_tensor(f"{base}_slst", TensorProto.INT64, [4], [0,0,0,0])
                graph.initializer.append(slice_starts)
                
                final_tensor = f"{base}_final"
                nodes_to_add.append(helper.make_node(
                    "Slice", [dq_tensor_4d, f"{base}_slst", orig_shape],
                    [final_tensor], name=f"/{base}/Unpad"
                ))
                input_replacements[inp_name] = final_tensor
            else:
                input_replacements[inp_name] = dq_tensor_4d
    
    # Update MatMul inputs
    for matmul in attention_matmuls:
        new_inputs = [input_replacements.get(inp, inp) for inp in matmul.input]
        matmul.ClearField('input')
        matmul.input.extend(new_inputs)
    
    graph.node.extend(nodes_to_add)
    
    print(f"\nSaving: {output_path}")
    onnx.save(model, output_path)
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    from collections import Counter
    ops = Counter(n.op_type for n in onnx.load(output_path).graph.node)
    print(f"\nTRT_FP4DynamicQuantize: {ops.get('TRT_FP4DynamicQuantize', 0)}")
    print(f"DequantizeLinear: {ops.get('DequantizeLinear', 0)}")
    print(f"Pad nodes: {ops.get('Pad', 0)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/workspace/models/vit_nvfp4_bs_064.onnx')
    parser.add_argument('--output', default='/workspace/models/vit_nvfp4_full.onnx')
    args = parser.parse_args()
    add_fp4_to_attention(args.input, args.output)
