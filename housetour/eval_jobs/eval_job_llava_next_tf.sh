#!/bin/bash
#SBATCH --partition=serc # Specify the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      
#SBATCH -C GPU_MEM:40GB
#SBATCH --mem=30GB
#SBATCH --time=30:00:00
#SBATCH --job-name=llava_next
#SBATCH --output=cluster_logs/%j/%j.out
#SBATCH --error=cluster_logs/%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /scratch/users/atacelen

ml load python/3.12.1
ml load cuda/12.4.0

source transfer/envs/llava/bin/activate

cd housetour

python3 eval_llava_next_tf.py