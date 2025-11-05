#!/bin/bash
#SBATCH --job-name=infer_vars33V
#SBATCH --ntasks-per-node=1
#SBATCH -A MST113255
#SBATCH --gres=gpu:1
#SBATCH --partition=dev
#SBATCH --exclude=hgpn17
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

#
# ---- settings
config_name=vars33_V
out_name=/work/choutilin1/out_vars33/vars33_V/all_XXXX.nc

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
	--run_num "01" \
	--mode="lite" \
	--samples_offset=XXXX

date
