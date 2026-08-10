#!/bin/bash
#SBATCH --job-name=infer_sfno
#SBATCH --ntasks-per-node=1
#SBATCH -A MST113255
#SBATCH --gres=gpu:2
#SBATCH --mem=1600G
#SBATCH --partition=8gpus
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --time=4:00:00
#
# ---- settings
config_name=earth2sfno
out_name="/work/choutilin1/out_vars103/earth2sfno/19990101.nc"

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
        --inference_output_path=$out_name \
	--overwrite_output_path=False \
	--run_num "00" \
	--mode="lite" \
	--samples_offset=0

date


