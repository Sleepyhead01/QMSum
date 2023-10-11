#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=126000M
#SBATCH --time=20:20:00
#SBATCH --account=def-fard
#SBATCH --output=infer_summ_train_longllama.txt
#SBATCH --mail-user=millon@student.ubc.ca

source /home/millon/projects/def-fard/millon/long/bin/activate

python generate.py
