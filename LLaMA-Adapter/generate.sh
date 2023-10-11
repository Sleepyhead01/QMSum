#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=126000M
#SBATCH --time=11:00:00
#SBATCH --account=def-fard
#SBATCH --output=direct_inference_500_1000_500_1000_.txt
#SBATCH --mail-user=millon@student.ubc.ca

source /home/millon/projects/def-fard/millon/LA/bin/activate

torchrun --nproc_per_node 1 generate_summary.py \
--ckpt_dir /home/millon/projects/def-fard/millon/LLaMA-Adapter/llama_adapter_v2_multimodal7b/models/7B \
--tokenizer_path /home/millon/projects/def-fard/millon/LLaMA-Adapter/llama_adapter_v2_multimodal7b/models/tokenizer.model \
--adapter_path /home/millon/projects/def-fard/millon/LLaMA-Adapter/alpaca_finetuning_v1/adapters/llama_adapter_len10_layer30_release.pth \
--quantizer False

