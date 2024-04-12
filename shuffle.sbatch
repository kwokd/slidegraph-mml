#!/bin/bash
#
#SBATCH --job-name=SHUFFLE_MMLSURV # Job name for tracking
#SBATCH --partition=falcon,gecko  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./batch_out/shuffle_%j.out # Standard out from your job
#SBATCH --error=./batch_out/shuffle_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA
conda activate proj
srun python mmlsurv_new.py -f "SHUFFLE"