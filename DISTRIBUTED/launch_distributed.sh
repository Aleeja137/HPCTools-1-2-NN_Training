#!/bin/bash -l
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32 
#SBATCH --mem=64G          
#SBATCH --gres=gpu:a100:2  
#SBATCH --time=00:30:00    

source $STORE/mypython/bin/deactivate
source $STORE/mypython/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
export WORLD_SIZE=$SLURM_NPROCS
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# export RANK=$SLURM_PROCID
# echo "RANK="$RANK


python -u distributed.py