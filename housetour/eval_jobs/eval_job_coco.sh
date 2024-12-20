#!/bin/bash
#SBATCH --partition=serc # Specify the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=10:00:00
#SBATCH --job-name=coco
#SBATCH --output=cluster_logs/%j/%j.out
#SBATCH --error=cluster_logs/%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /scratch/users/atacelen

ml load python
ml load java
pip install numpy

cd coco-caption/pycocoevalcap

python coco_eval.py --capt_path "/scratch/users/atacelen/housetour/eval_qwen2_capt.jsonl" --name qwen2_coco
