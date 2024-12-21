#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=16G
#SBATCH -c 32
#SBATCH --time=01:30:00

# Load and activate your Python environment
source $STORE/mypython/bin/activate
python -m ipykernel install --user --name mypython --display-name "Python (mypython)"

# Verify the Python path and run the Python script
which python

# Run the notebook and execute all cells
# jupyter nbconvert --to notebook --execute baseline.ipynb --inplace
# jupyter nbconvert --to notebook --execute baseline.ipynb --inplace --ExecutePreprocessor.kernel_name=mypython
jupyter-nbconvert --to notebook --execute baseline.ipynb --inplace --ExecutePreprocessor.kernel_name=mypython --ExecutePreprocessor.timeout=-1 # Loading these heavy libraries takes time, disable cell timeout
