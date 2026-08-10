#!/bin/bash
#SBATCH --job-name=infer_
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=28
#SBATCH --partition=p07
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

#
# ---- settings

date

#srun --mpi=pmix --container-image /mnt/shared/p07/makani-torch2410-v2.1.sqsh --container-remap-root --container-mounts /mnt/shared/p07:/mnt/shared/p07,/mnt/home/choutilin-ntu_-efeb0f/:/mnt/home/choutilin-ntu_-efeb0f/ /mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/run_inferenceJan01.sh

#srun --mpi=pmix --container-image /mnt/shared/p07/makani-torch2410-v2.1.sqsh --container-remap-root --container-mounts /mnt/shared/p07:/mnt/shared/p07,/mnt/home/choutilin-ntu_-efeb0f/:/mnt/home/choutilin-ntu_-efeb0f/ /mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/run_inferenceApr01.sh

#srun --mpi=pmix --container-image /mnt/shared/p07/makani-torch2410-v2.1.sqsh --container-remap-root --container-mounts /mnt/shared/p07:/mnt/shared/p07,/mnt/home/choutilin-ntu_-efeb0f/:/mnt/home/choutilin-ntu_-efeb0f/ /mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/run_inferenceJul01.sh

srun --mpi=pmix --container-image /mnt/shared/p07/makani-torch2410-v2.1.sqsh --container-remap-root --container-mounts /mnt/shared/p07:/mnt/shared/p07,/mnt/home/choutilin-ntu_-efeb0f/:/mnt/home/choutilin-ntu_-efeb0f/ /mnt/home/choutilin-ntu_-efeb0f/vars103_BG/v250508/run_inferenceOct01.sh

date


