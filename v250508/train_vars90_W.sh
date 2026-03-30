#!/bin/bash
#SBATCH --job-name=train_vars90_W
#SBATCH -A MST113255
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=choutilin@ntu.edu.tw

#
# ---- settings
config_name=vars90_W


date
TORCHRUN=/work/choutilin1/.conda/envs/makani/bin/torchrun
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "TORCHRUN=$TORCHRUN"
#nvidia-smi -l 10 &
#nsys profile --trace cuda,nvtx,osrt --output /work/choutilin1/baseline --force-overwrite true --cuda-memory-usage true \
$TORCHRUN --master_port=$FREE_PORT \
    --nnodes=1                     \
    --nproc_per_node=1             \
    /home/choutilin1/makani/makani/train.py                \
        --yaml_config="config/sfnonet.yaml" \
        --config=$config_name \
	--run_num "00"
#	--amp_mode=bf16       \

date


