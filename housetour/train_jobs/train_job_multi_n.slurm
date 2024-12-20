#!/bin/bash
#SBATCH --partition=serc # Specify the GPU partition
#SBATCH --nodes=8
#SBATCH --gpus-per-task=1
#SBATCH -C GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:80GB
#SBATCH --mem=60GB
#SBATCH --time=5:00:00
#SBATCH --job-name=timechat_first_train
#SBATCH --output=cluster_logs/%j/%j.out
#SBATCH --error=cluster_logs/%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export LAUNCHER=" \
torchrun \
--nnodes 8 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
" 
export PYTHON_FILE="/scratch/users/atacelen/housetour/train_dist.py"
export ARGS="--cfg_path grandtour/configs/grandtour_train.yaml"
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"

cd /scratch/users/atacelen

ml load python/3.12.1

source transfer/envs/grandtour/bin/activate

cd housetour

srun --export=ALL $CMD
