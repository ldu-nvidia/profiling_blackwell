#!/bin/bash
# Create minimal transfer package for Blackwell

PACKAGE_NAME="profiling_blackwell_$(date +%Y%m%d)"
PACKAGE_DIR="/tmp/$PACKAGE_NAME"

mkdir -p "$PACKAGE_DIR"

echo "Creating transfer package..."

# Essential files
cp -v *.onnx "$PACKAGE_DIR/"           # ONNX models (~301M)
cp -v *.sh "$PACKAGE_DIR/"             # Shell scripts
cp -v *.txt "$PACKAGE_DIR/"            # requirements.txt
cp -rv utils/ "$PACKAGE_DIR/"          # Python utilities
cp -rv logs/ "$PACKAGE_DIR/"           # Logs (for reference)

# Optionally include H100 results for comparison
mkdir -p "$PACKAGE_DIR/results_h100"
cp -rv results/ "$PACKAGE_DIR/results_h100/"

# Create setup instructions
cat > "$PACKAGE_DIR/SETUP_BLACKWELL.md" << 'SETUP'
# Setup on Blackwell

## 1. Check for available tools
```bash
which trtexec    # TensorRT
which nsys       # Nsight Systems  
which ncu        # Nsight Compute
```

## 2. Create Python environment
```bash
python3 -m venv profiling_env
source profiling_env/bin/activate
pip install -r requirements.txt
```

## 3. If TensorRT not available, check module system
```bash
module avail tensorrt
module load tensorrt
```

## 4. Run profiling
```bash
# Get a GPU node
srun -p interactive --gres=gpu:1 --time=02:00:00 --pty bash

# Run trtexec profiling
./run_trtexec_profiling.sh
```
SETUP

# Create tarball
cd /tmp
tar -czvf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"

echo ""
echo "Package created: /tmp/${PACKAGE_NAME}.tar.gz"
ls -lh "/tmp/${PACKAGE_NAME}.tar.gz"
echo ""
echo "Transfer with:"
echo "  scp /tmp/${PACKAGE_NAME}.tar.gz <blackwell-host>:~/"
echo ""
echo "On Blackwell:"
echo "  tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  cd ${PACKAGE_NAME}"

