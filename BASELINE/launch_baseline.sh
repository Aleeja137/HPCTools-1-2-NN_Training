#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=16G
#SBATCH -c 32
#SBATCH --time=01:30:00

# Load and activate your Python environment
source $STORE/mypython/bin/activate

# Use srun to distribute the Python script execution across the resources
srun python baseline.py
