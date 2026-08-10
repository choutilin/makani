#!/bin/bash
#SBATCH --job-name=infer_sfno
#SBATCH --ntasks-per-node=1
#SBATCH -A MST113255
#SBATCH --gres=gpu:1
#SBATCH --partition=dev
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#
# ---- settings
config_name=earth2sfno

date
TORCHRUN=/work/choutilin1/.conda/envs/makani/bin/torchrun
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

echo "TORCHRUN=$TORCHRUN"
$TORCHRUN --master_port=$FREE_PORT \
    --nnodes=1                     \
    --nproc_per_node=1             \
    /home/choutilin1/makani/makani/inference.py                \
        --yaml_config="config/sfnonet.yaml" \
        --config=$config_name \
	--run_num "00" \
	--mode="score" \

date


