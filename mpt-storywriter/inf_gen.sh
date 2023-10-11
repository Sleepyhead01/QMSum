#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=24         # CPU cores/threads
#SBATCH --mem=300000M
#SBATCH --time=1:00:00
#SBATCH --account=def-fard
#SBATCH --output=results/inference_results_mpt_inf_mcpu.txt
#SBATCH --mail-user=millon@student.ubc.ca


source /home/millon/projects/def-fard/millon/LA/bin/activate

python mpt_generate.py
