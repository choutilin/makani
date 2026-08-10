#!/bin/bash
# ---- settings
config_name=vars103_BG



date

MAKANIDIR=/mnt/home/choutilin-ntu_-efeb0f/vars103_BG/makani/
cd $MAKANIDIR


export NUMBA_DISABLE_CUDA=1


FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "TORCHRUN=$TORCHRUN"
#nvidia-smi -l 10 &
#nsys profile --trace cuda,nvtx,osrt --output /work/choutilin1/baseline --force-overwrite true --cuda-memory-usage true \
export PYTHONUSERBASE=$MAKANIDIR/.local
mkdir $PYTHONUSERBASE


export TRITON_LIBCUDA_PATH=/.singularity.d/libs/libcuda.so.1
#pip install -e .
out_name="/mnt/shared/p07/choutilin1/out_vars103/vars103_BG/20250701.nc"


export TRITON_LIBCUDA_PATH=/.singularity.d/libs/libcuda.so.1
torchrun --master_port=$FREE_PORT \
    --nnodes=1                     \
    --nproc_per_node=1             \
    ./makani/inference.py                \
        --yaml_config="/mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/config/sfnonet.yaml" \
        --config=$config_name \
	--inference_output_path=$out_name \
	--overwrite_output_path=False \
	--mode="lite" \
	--run_num "00" \
	--samples_offset=724

date


