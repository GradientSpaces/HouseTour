#!/bin/bash
#SBATCH --partition=serc # Specify the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      
#SBATCH -C GPU_MEM:80GB
#SBATCH --mem=30GB
#SBATCH --time=20:00:00
#SBATCH --job-name=timechat_first_train
#SBATCH --output=cluster_logs/%j/%j.out
#SBATCH --error=cluster_logs/%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /scratch/users/atacelen

ml load python/3.12.1
ml load cuda/12.4.0

source transfer/envs/grandtour/bin/activate

cd housetour

python3 eval_full.py