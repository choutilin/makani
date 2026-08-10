#!/bin/bash
#SBATCH --job-name=training
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=192
#SBATCH --partition=p07
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --mail-user=choutilin@gmail.com
#SBATCH --mail-type=FAIL

#
# ---- settings

date


srun --mpi=pmix --container-image /mnt/shared/p07/makani-torch2410-v2.1.sqsh --container-remap-root --container-mounts /mnt/shared/p07:/mnt/shared/p07,/mnt/home/choutilin-ntu_-efeb0f/:/mnt/home/choutilin-ntu_-efeb0f/ /mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/run_training.sh

date


