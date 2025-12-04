#!/bin/bash
# Nsight Systems wrapper with local cuDNN
PROFILING_DIR="/lustre/fsw/portfolios/general/users/ldu/profiling"
export LD_LIBRARY_PATH=/lustre/fsw/portfolios/general/users/ldu/profiling/tensorrt/lib:/lustre/fsw/portfolios/general/users/ldu/profiling/cudnn/lib:$LD_LIBRARY_PATH
$PROFILING_DIR/nsight-systems/bin/nsys "$@"
