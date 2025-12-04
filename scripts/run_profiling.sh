#!/bin/bash
################################################################################
# TensorRT Profiling with Nsight Tools
#
# Profiles ViT models (FP16, MXFP8, NVFP4) using:
#   - Nsight Systems (nsys) - System-wide GPU timeline
#   - Nsight Compute (ncu) - Kernel-level metrics
#   - TensorRT trtexec - Engine building and benchmarking
#
# Uses Docker container: nvcr.io/nvidia/pytorch:25.06-py3
# (Contains TensorRT, Nsight tools, and ModelOpt for MXFP8 support)
#
# Usage:
#   ./scripts/run_profiling.sh              # Run all profiling
#   ./scripts/run_profiling.sh --nsys       # Nsight Systems only
#   ./scripts/run_profiling.sh --ncu        # Nsight Compute only
#   ./scripts/run_profiling.sh --benchmark  # Benchmark only (no profiling)
#   ./scripts/run_profiling.sh --help       # Show help
#
################################################################################

set -e

# Get script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
WARMUP=50
ITERATIONS=100
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Directories (relative to PROJECT_ROOT)
MODELS_DIR="models"
ENGINES_DIR="engines"
LOGS_DIR="logs"

# Results organized by run timestamp
RUN_DIR="results/runs/${TIMESTAMP}"
NSYS_DIR="$RUN_DIR/nsight-systems"
NCU_DIR="$RUN_DIR/nsight-compute"
BENCH_DIR="$RUN_DIR/benchmark"

# Create directories
mkdir -p "$PROJECT_ROOT/$NSYS_DIR"
mkdir -p "$PROJECT_ROOT/$NCU_DIR"
mkdir -p "$PROJECT_ROOT/$BENCH_DIR"
mkdir -p "$PROJECT_ROOT/$ENGINES_DIR"
mkdir -p "$PROJECT_ROOT/$LOGS_DIR"

LOG_FILE="$PROJECT_ROOT/$LOGS_DIR/profiling_${TIMESTAMP}.log"

# Models to profile (MXFP8 disabled - requires TRT_MXFP8* plugins not in container)
declare -a MODELS=(
    "vit_fp16_bs_064.onnx:fp16"
    # "vit_mxfp8_bs_064.onnx:mxfp8"  # Disabled: Missing TRT_MXFP8DequantizeLinear plugin
    "vit_nvfp4_bs_064.onnx:nvfp4"
)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

log() {
    # Write to stderr (fd 2) so it doesn't interfere with function return values via stdout
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

show_help() {
    echo "TensorRT Profiling with Nsight Tools"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_profiling.sh              Run all profiling (nsys + ncu + benchmark)"
    echo "  ./scripts/run_profiling.sh --nsys       Nsight Systems profiling only"
    echo "  ./scripts/run_profiling.sh --ncu        Nsight Compute profiling only"
    echo "  ./scripts/run_profiling.sh --benchmark  Benchmark only (no profiling overhead)"
    echo "  ./scripts/run_profiling.sh --build      Build engines only"
    echo "  ./scripts/run_profiling.sh --help       Show this help"
    echo ""
    echo "Models profiled:"
    echo "  - vit_fp16_bs_064.onnx   (FP16 baseline)"
    echo "  - vit_mxfp8_bs_064.onnx  (MXFP8 - requires ModelOpt)"
    echo "  - vit_nvfp4_bs_064.onnx  (NVFP4)"
    echo ""
    echo "Container: $CONTAINER_IMAGE"
    echo "Results:   $PROJECT_ROOT/$RUN_DIR/"
}

check_environment() {
    log "${BLUE}>>> Checking environment...${NC}"
    
    # Check GPU
    if ! nvidia-smi &>/dev/null; then
        log "${RED}ERROR: No GPU detected!${NC}"
        exit 1
    fi
    
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "GPU: $GPU_NAME (x$GPU_COUNT)"
    
    # Check Docker
    if ! docker info &>/dev/null; then
        log "${RED}ERROR: Docker not available!${NC}"
        exit 1
    fi
    log "Docker: $(docker --version | cut -d' ' -f3)"
    
    # Check container image
    if ! docker image inspect "$CONTAINER_IMAGE" &>/dev/null; then
        log "${YELLOW}Pulling container image: $CONTAINER_IMAGE${NC}"
        docker pull "$CONTAINER_IMAGE"
    fi
    log "Container: $CONTAINER_IMAGE"
    
    # Check models
    for model_config in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_FILE _ <<< "$model_config"
        if [[ ! -f "$PROJECT_ROOT/$MODELS_DIR/$MODEL_FILE" ]]; then
            log "${RED}ERROR: Model not found: $MODELS_DIR/$MODEL_FILE${NC}"
            exit 1
        fi
    done
    log "Models: All found in $MODELS_DIR/"
    
    echo ""
}

# Docker run helper - runs a command inside container
# Usage: run_docker "command" "output_file" [privileged]
run_docker() {
    local CMD="$1"
    local OUTPUT_FILE="$2"
    local USE_PRIVILEGED="${3:-false}"
    
    if [[ "$USE_PRIVILEGED" == "true" ]]; then
        docker run --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            --privileged --cap-add=SYS_ADMIN \
            -v "$PROJECT_ROOT:/workspace/profiling" \
            -w /workspace/profiling \
            "$CONTAINER_IMAGE" \
            bash -c "$CMD" > "$OUTPUT_FILE" 2>&1
    else
        docker run --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "$PROJECT_ROOT:/workspace/profiling" \
            -w /workspace/profiling \
            "$CONTAINER_IMAGE" \
            bash -c "$CMD" > "$OUTPUT_FILE" 2>&1
    fi
    
    return $?
}

################################################################################
# Engine Building
################################################################################

build_engine() {
    local MODEL_FILE=$1
    local MODEL_SHORT=$2
    local ENGINE_FILE="$ENGINES_DIR/${MODEL_SHORT}_${TIMESTAMP}.engine"
    local BUILD_LOG="$PROJECT_ROOT/$ENGINES_DIR/${MODEL_SHORT}_${TIMESTAMP}_build.log"
    
    log "Building TensorRT engine: $MODEL_SHORT"
    
    # MXFP8 models require TRT_MXFP8DequantizeLinear plugin which is not available
    # in standard TensorRT - skip with warning
    if [[ "$MODEL_SHORT" == "mxfp8" ]]; then
        log "${YELLOW}⚠ MXFP8 model requires TRT_MXFP8DequantizeLinear plugin${NC}"
        log "${YELLOW}  This plugin is not available in the current container.${NC}"
        log "${YELLOW}  Skipping MXFP8 - to enable, use a container with ModelOpt TRT plugins.${NC}"
        return 1
    fi
    
    # Build command based on precision
    local BUILD_FLAGS=""
    case "$MODEL_SHORT" in
        fp16)
            BUILD_FLAGS="--fp16"
            ;;
        nvfp4)
            # NVFP4 models have quantization baked in, use stronglyTyped
            BUILD_FLAGS="--fp16 --stronglyTyped"
            ;;
    esac
    
    # Run trtexec in container - pass command as separate args (not bash -c)
    run_docker "trtexec --onnx=/workspace/profiling/$MODELS_DIR/$MODEL_FILE --saveEngine=/workspace/profiling/$ENGINE_FILE $BUILD_FLAGS --verbose" \
        "$BUILD_LOG" "false"
    
    if [[ -f "$PROJECT_ROOT/$ENGINE_FILE" ]]; then
        local SIZE=$(du -h "$PROJECT_ROOT/$ENGINE_FILE" | cut -f1)
        log "${GREEN}✓ Engine built: $ENGINE_FILE ($SIZE)${NC}"
        echo "$ENGINE_FILE"
        return 0
    fi
    
    log "${RED}✗ Engine build failed for $MODEL_SHORT${NC}"
    log "${RED}  Check log: $BUILD_LOG${NC}"
    return 1
}

################################################################################
# Nsight Systems Profiling
################################################################################

profile_nsys() {
    local ENGINE_FILE=$1
    local MODEL_SHORT=$2
    local OUTPUT_DIR="$NSYS_DIR/${MODEL_SHORT}"
    
    mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"
    
    log ""
    log "${BLUE}>>> Nsight Systems profiling: $MODEL_SHORT${NC}"
    log "    Output: $OUTPUT_DIR"
    
    # Build command as single line
    local CMD="nsys profile --output=/workspace/profiling/$OUTPUT_DIR/profile --force-overwrite=true --trace=cuda,nvtx --cuda-memory-usage=true --stats=true trtexec --loadEngine=/workspace/profiling/$ENGINE_FILE --warmUp=$WARMUP --iterations=$ITERATIONS --useCudaGraph --noDataTransfers"
    
    # Run with privileged mode for nsys
    run_docker "$CMD" "$PROJECT_ROOT/$OUTPUT_DIR/profiling.log" "true"
    
    if [[ -f "$PROJECT_ROOT/$OUTPUT_DIR/profile.nsys-rep" ]]; then
        log "${GREEN}✓ Nsight Systems profiling complete${NC}"
        
        # Generate stats report
        run_docker "nsys stats /workspace/profiling/$OUTPUT_DIR/profile.nsys-rep --report gputrace" \
            "$PROJECT_ROOT/$OUTPUT_DIR/gpu_trace.txt" "true" || true
            
        return 0
    else
        log "${YELLOW}⚠ Nsight Systems profiling had issues - check $OUTPUT_DIR/profiling.log${NC}"
        return 1
    fi
}

################################################################################
# Nsight Compute Profiling
################################################################################

profile_ncu() {
    local ENGINE_FILE=$1
    local MODEL_SHORT=$2
    local OUTPUT_DIR="$NCU_DIR/${MODEL_SHORT}"
    
    mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"
    
    log ""
    log "${BLUE}>>> Nsight Compute profiling: $MODEL_SHORT${NC}"
    log "    Output: $OUTPUT_DIR"
    log "    Note: NCU is slow - profiling limited kernel launches"
    
    # Build command as single line
    local CMD="ncu --output /workspace/profiling/$OUTPUT_DIR/profile --force-overwrite --set full --launch-skip 10 --launch-count 5 trtexec --loadEngine=/workspace/profiling/$ENGINE_FILE --warmUp=5 --iterations=20 --noDataTransfers"
    
    # Run with privileged mode for ncu
    run_docker "$CMD" "$PROJECT_ROOT/$OUTPUT_DIR/profiling.log" "true"
    
    if [[ -f "$PROJECT_ROOT/$OUTPUT_DIR/profile.ncu-rep" ]]; then
        log "${GREEN}✓ Nsight Compute profiling complete${NC}"
        
        # Export to CSV
        run_docker "ncu --import /workspace/profiling/$OUTPUT_DIR/profile.ncu-rep --csv" \
            "$PROJECT_ROOT/$OUTPUT_DIR/metrics.csv" "true" || true
            
        return 0
    else
        log "${YELLOW}⚠ Nsight Compute profiling had issues - check $OUTPUT_DIR/profiling.log${NC}"
        return 1
    fi
}

################################################################################
# Benchmark (no profiling overhead)
################################################################################

run_benchmark() {
    local ENGINE_FILE=$1
    local MODEL_SHORT=$2
    local OUTPUT_DIR="$BENCH_DIR"
    
    mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"
    
    log ""
    log "${BLUE}>>> Benchmarking: $MODEL_SHORT${NC}"
    
    # Build command as single line
    local CMD="trtexec --loadEngine=/workspace/profiling/$ENGINE_FILE --warmUp=$WARMUP --iterations=$ITERATIONS --useCudaGraph --noDataTransfers"
    
    local OUTPUT_FILE="$PROJECT_ROOT/$OUTPUT_DIR/${MODEL_SHORT}.log"
    run_docker "$CMD" "$OUTPUT_FILE" "false"
    
    # Parse metrics from output
    local LAT_MEAN=$(grep -oP 'mean = \K[\d.]+' "$OUTPUT_FILE" | head -1 || echo "0")
    local LAT_MEDIAN=$(grep -oP 'median = \K[\d.]+' "$OUTPUT_FILE" | head -1 || echo "0")
    local LAT_P99=$(grep -oP 'percentile\(99%\) = \K[\d.]+' "$OUTPUT_FILE" | head -1 || echo "0")
    local THROUGHPUT=$(grep -oP 'Throughput: \K[\d.]+' "$OUTPUT_FILE" | head -1 || echo "0")
    
    # Validate we got values
    [[ -z "$LAT_MEAN" || "$LAT_MEAN" == "" ]] && LAT_MEAN="0"
    [[ -z "$LAT_MEDIAN" || "$LAT_MEDIAN" == "" ]] && LAT_MEDIAN="0"
    [[ -z "$LAT_P99" || "$LAT_P99" == "" ]] && LAT_P99="0"
    [[ -z "$THROUGHPUT" || "$THROUGHPUT" == "" ]] && THROUGHPUT="0"
    
    log "    Mean Latency:   ${LAT_MEAN} ms"
    log "    Median Latency: ${LAT_MEDIAN} ms"
    log "    P99 Latency:    ${LAT_P99} ms"
    log "    Throughput:     ${THROUGHPUT} qps"
    
    # Save metrics JSON
    cat > "$PROJECT_ROOT/$OUTPUT_DIR/${MODEL_SHORT}.json" << EOF
{
    "model": "$MODEL_SHORT",
    "engine_file": "$ENGINE_FILE",
    "timestamp": "$TIMESTAMP",
    "warmup_iterations": $WARMUP,
    "benchmark_iterations": $ITERATIONS,
    "latency_mean_ms": $LAT_MEAN,
    "latency_median_ms": $LAT_MEDIAN,
    "latency_p99_ms": $LAT_P99,
    "throughput_qps": $THROUGHPUT
}
EOF
    
    # Store for report generation (replace - with _ for variable names)
    local VAR_NAME=$(echo "$MODEL_SHORT" | tr '-' '_')
    eval "RESULT_${VAR_NAME}_MEAN=$LAT_MEAN"
    eval "RESULT_${VAR_NAME}_P99=$LAT_P99"
    eval "RESULT_${VAR_NAME}_THROUGHPUT=$THROUGHPUT"
}

################################################################################
# Report Generation
################################################################################

generate_report() {
    local REPORT_FILE="$PROJECT_ROOT/$RUN_DIR/REPORT.txt"
    
    log ""
    log "${BLUE}>>> Generating comparison report${NC}"
    
    {
        echo "================================================================================"
        echo "TENSORRT PROFILING REPORT"
        echo "================================================================================"
        echo "Generated: $(date)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "Container: $CONTAINER_IMAGE"
        echo ""
        echo "Configuration:"
        echo "  Warmup: $WARMUP iterations"
        echo "  Benchmark: $ITERATIONS iterations"
        echo ""
        echo "================================================================================"
        echo "LATENCY COMPARISON"
        echo "================================================================================"
        echo ""
        printf "%-12s %15s %15s %15s\n" "Model" "Mean (ms)" "P99 (ms)" "Throughput (qps)"
        echo "--------------------------------------------------------------------------------"
        
        for model_config in "${MODELS[@]}"; do
            IFS=':' read -r _ MODEL_SHORT <<< "$model_config"
            local MEAN_VAR="RESULT_${MODEL_SHORT}_MEAN"
            local P99_VAR="RESULT_${MODEL_SHORT}_P99"
            local THRU_VAR="RESULT_${MODEL_SHORT}_THROUGHPUT"
            printf "%-12s %15s %15s %15s\n" "$MODEL_SHORT" "${!MEAN_VAR:-N/A}" "${!P99_VAR:-N/A}" "${!THRU_VAR:-N/A}"
        done
        
        echo ""
        echo "================================================================================"
        echo "SPEEDUP vs FP16 BASELINE"
        echo "================================================================================"
        echo ""
        
        if [[ -n "$RESULT_fp16_MEAN" ]] && [[ "$RESULT_fp16_MEAN" != "N/A" ]]; then
            for model_config in "${MODELS[@]}"; do
                IFS=':' read -r _ MODEL_SHORT <<< "$model_config"
                if [[ "$MODEL_SHORT" != "fp16" ]]; then
                    local MEAN_VAR="RESULT_${MODEL_SHORT}_MEAN"
                    if [[ -n "${!MEAN_VAR}" ]] && [[ "${!MEAN_VAR}" != "N/A" ]]; then
                        local SPEEDUP=$(echo "scale=2; $RESULT_fp16_MEAN / ${!MEAN_VAR}" | bc 2>/dev/null || echo "N/A")
                        echo "  $MODEL_SHORT vs FP16: ${SPEEDUP}x speedup"
                    fi
                fi
            done
        fi
        
        echo ""
        echo "================================================================================"
        echo "OUTPUT FILES"
        echo "================================================================================"
        echo ""
        echo "Nsight Systems:"
        ls -la "$PROJECT_ROOT/$NSYS_DIR/" 2>/dev/null | tail -10 || echo "  None"
        echo ""
        echo "Nsight Compute:"
        ls -la "$PROJECT_ROOT/$NCU_DIR/" 2>/dev/null | tail -10 || echo "  None"
        echo ""
        echo "Engines:"
        ls -lh "$PROJECT_ROOT/$ENGINES_DIR/"*_${TIMESTAMP}.engine 2>/dev/null || echo "  None"
        echo ""
        
    } | tee "$REPORT_FILE"
    
    log ""
    log "Report saved: $REPORT_FILE"
}

################################################################################
# Main Profile Function
################################################################################

profile_model() {
    local MODEL_FILE=$1
    local MODEL_SHORT=$2
    local MODE=$3
    
    log ""
    log "================================================================================"
    log "PROFILING: $MODEL_SHORT"
    log "================================================================================"
    
    # Build engine first
    local ENGINE_FILE
    ENGINE_FILE=$(build_engine "$MODEL_FILE" "$MODEL_SHORT")
    if [[ $? -ne 0 ]] || [[ -z "$ENGINE_FILE" ]]; then
        log "${RED}Skipping $MODEL_SHORT - engine build failed${NC}"
        return 1
    fi
    
    # Run requested profiling
    case "$MODE" in
        nsys)
            profile_nsys "$ENGINE_FILE" "$MODEL_SHORT"
            ;;
        ncu)
            profile_ncu "$ENGINE_FILE" "$MODEL_SHORT"
            ;;
        benchmark)
            run_benchmark "$ENGINE_FILE" "$MODEL_SHORT"
            ;;
        build)
            # Engine already built
            ;;
        all|*)
            profile_nsys "$ENGINE_FILE" "$MODEL_SHORT" || true
            profile_ncu "$ENGINE_FILE" "$MODEL_SHORT" || true
            run_benchmark "$ENGINE_FILE" "$MODEL_SHORT"
            ;;
    esac
    
    log "${GREEN}✓ $MODEL_SHORT complete${NC}"
}

################################################################################
# Main
################################################################################

main() {
    local MODE="${1:-all}"
    
    case "$MODE" in
        --help|-h)
            show_help
            exit 0
            ;;
        --nsys)
            MODE="nsys"
            ;;
        --ncu)
            MODE="ncu"
            ;;
        --benchmark)
            MODE="benchmark"
            ;;
        --build)
            MODE="build"
            ;;
        *)
            MODE="all"
            ;;
    esac
    
    log "================================================================================"
    log "TENSORRT PROFILING - FP16 / MXFP8 / NVFP4"
    log "================================================================================"
    log "Timestamp: $TIMESTAMP"
    log "Mode: $MODE"
    log "Log: $LOG_FILE"
    log ""
    
    check_environment
    
    local SUCCESSFUL=0
    local FAILED=0
    
    for model_config in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_FILE MODEL_SHORT <<< "$model_config"
        
        if profile_model "$MODEL_FILE" "$MODEL_SHORT" "$MODE"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done
    
    # Generate report if we ran benchmarks
    if [[ "$MODE" == "all" ]] || [[ "$MODE" == "benchmark" ]]; then
        generate_report
    fi
    
    log ""
    log "================================================================================"
    log "PROFILING COMPLETE"
    log "================================================================================"
    log "Successful: $SUCCESSFUL / ${#MODELS[@]}"
    log "Failed: $FAILED / ${#MODELS[@]}"
    log "Results: $PROJECT_ROOT/$RUN_DIR/"
    log "Engines: $PROJECT_ROOT/$ENGINES_DIR/"
    log "Log: $LOG_FILE"
    log "================================================================================"
}

# Run
main "$@"

