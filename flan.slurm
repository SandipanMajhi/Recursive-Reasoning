#!/bin/bash
#
#SBATCH --job-name=flv
#SBATCH --partition=brtx6-ir
#SBATCH --gpus=1

python3 flan_inference_deduction.py --dataset_name strategyqa --mode tilde --seed 42