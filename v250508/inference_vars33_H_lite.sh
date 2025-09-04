#!/bin/bash
#SBATCH --job-name=infer_vars33H
#SBATCH --ntasks-per-node=1
#SBATCH -A MST112107
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=choutilin@ntu.edu.tw

#
# ---- settings
config_name=vars33_H
out_name=/work/choutilin1/out_vars33/vars33_H/out_0000.nc

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
	--run_num "04" \
	--mode="lite"

date

# #module load openmpi/4.1.6_ucx1.14.1_cuda12.3
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) # a large num
# echo "MASTER_ADDR="$MASTER_ADDR
# echo "WORLD_SIZE="$WORLD_SIZE
# echo "LOCAL_RANK="$LOCAL_RANK

#
#
#
# # ---- for modulus distributed (MPI)
# export OMP_NUM_THREADS=1
# export WORLD_SIZE=1 # the number of GPUs
# export RANK=0

#module load cuda/10.2
# partition: gtest, gp1d, gp2d, gp4d

# 要用較多gpu的話可以用gp1d~4d (可以跑的時長不一樣)
# 指令
#
# 加載環境
# sbatch start_train.sh


# 確認可用的partition
# sinfo -s



# some try

#CUDA_LAUNCH_BLOCKING=1 python3.11 train.py --yaml_config custom.yaml --config full_field

#./globusconnectpersonal -setup --no-gui


#Input a value for the Endpoint Name: MYTWCC
#registered new endpoint, id: 72751bf6-b235-11ef-a0e3-7754bd4249c4

#Input a value for the Endpoint Name: MYTWCC
#registered new endpoint, id: d47e1342-b236-11ef-8888-f349fadc53c5
#setup completed successfully



#將 Ubuntu 設置為端點後，啟用端點：
#globus endpoint activate 72751bf6-b235-11ef-a0e3-7754bd4249c4

#globus transfer 945b3c9e-0f8c-11ed-8daf-9f359c660fbd:/data/FCN_ERA5_data_v0/train/2015.h5 "72751bf6-b235-11ef-a0e3-7754bd4249c4:/work/s2024419/FourCastNet"
