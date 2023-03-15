#!/bin/bash
#
#SBATCH --job-name=recexp
#SBATCH --partition=brtx6-dev
#SBATCH --gpus=1

python3 main_gen2.py --dataset_name com2sense --mode normal --seed 21