#!/bin/bash
#
#SBATCH --job-name=recexp
#SBATCH --partition=brtx6
#SBATCH --gpus=1

python3 regenerate_data.py