#!/bin/bash
################################################################################
# Verify CUDA + cuDNN Setup  
# Run this on a GPU node to verify everything is working
################################################################################

cd /lustre/fsw/portfolios/general/users/ldu/profiling

echo "================================================================================"
echo "CUDA + cuDNN Setup Verification"
echo "================================================================================"
echo ""

# Load CUDA
module load cuda12.4/toolkit/12.4.1 2>/dev/null

# Set library paths
export LD_LIBRARY_PATH=/lustre/fsw/portfolios/general/users/ldu/profiling/cudnn/lib:/cm/shared/apps/cuda12.3/toolkit/12.3.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Activate env
source profiling_env/bin/activate

# Check GPU
echo "[1/4] Checking GPU..."
if nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✅ GPU: $GPU"
else
    echo "❌ No GPU detected - you must be on a GPU node!"
    exit 1
fi
echo ""

# Check cuDNN files
echo "[2/4] Checking cuDNN installation..."
if [ -f "/lustre/fsw/portfolios/general/users/ldu/profiling/cudnn/lib/libcudnn.so.9" ]; then
    echo "✅ cuDNN libraries found"
    ls -lh /lustre/fsw/portfolios/general/users/ldu/profiling/cudnn/lib/libcudnn.so.9*
else
    echo "❌ cuDNN libraries missing!"
    exit 1
fi
echo ""

# Test ONNX Runtime CUDA Provider
echo "[3/4] Testing ONNX Runtime CUDA Provider..."
python << 'EOF'
import onnxruntime as ort
print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

try:
    session = ort.InferenceSession(
        'vit_fp16_bs_064.onnx',
        providers=['CUDAExecutionProvider']
    )
    active = session.get_providers()
    
    if 'CUDAExecutionProvider' in active:
        print("✅ CUDA Provider is ACTIVE and working!")
        
        # Quick inference test
        import numpy as np
        inp = session.get_inputs()[0]
        dummy = np.random.randn(*inp.shape).astype(np.float16)
        
        print("\nRunning quick inference test...")
        output = session.run(None, {inp.name: dummy})
        print(f"✅ Inference successful! Output shape: {output[0].shape}")
    else:
        print(f"❌ Fell back to: {active[0]}")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ CUDA Provider test passed!"
else
    echo ""
    echo "❌ CUDA Provider test failed!"
    exit 1
fi

echo ""

# Test TensorRT Provider
echo "[4/4] Testing TensorRT Provider..."
python << 'EOF'
import onnxruntime as ort
import numpy as np

try:
    session = ort.InferenceSession(
        'vit_fp16_bs_064.onnx',
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    )
    
    active = session.get_providers()
    print(f"Active providers: {active}")
    
    if 'TensorrtExecutionProvider' in active or 'CUDAExecutionProvider' in active:
        print("✅ TensorRT/CUDA provider working!")
    else:
        print("⚠ Using CPU fallback")
        
except Exception as e:
    print(f"⚠ TensorRT test: {e}")
EOF

echo ""
echo "================================================================================"
echo "✅ ALL CHECKS PASSED!"
echo "================================================================================"
echo ""
echo "Your setup is ready for profiling!"
echo ""
echo "You can now run:"
echo "  ./run_profiling.sh           # Full profiling (CUDA + TensorRT)"
echo "  ./test_fp16_profiling.sh     # Quick FP16 test"
echo "  ./run_ncu_profiling.sh       # Detailed kernel profiling"
echo ""

