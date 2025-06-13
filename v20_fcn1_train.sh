#!/bin/bash
#SBATCH --job-name=v20_fcn1_y4a
#SBATCH --ntasks-per-node=4
#SBATCH -A MST112107
#SBATCH --gres=gpu:4
#SBATCH --partition=normal

date
TORCHRUN=/work/lkkbox945/models/makani/env/bin/torchrun
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

echo "TORCHRUN=$TORCHRUN"
$TORCHRUN --master_port=$FREE_PORT \
    --nnodes=1 \
    --nproc_per_node=4 \
    makani/train.py \
        --yaml_config="config/sfnonet.yaml" \
        --config="v20_fcn1_y4a" \
        --batch_size=4


echo python exited
date

