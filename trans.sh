#!/bin/bash
#SBATCH --job-name=item2vectrans
#SBATCH --output=1my-output.log
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=15-24:00:00
#SBATCH --cpus-per-task=30
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=hpc03


# mkdir -p /hpc/scratch/$user
# load modules



# module load cuda11.2/blas/11.2.2
# module load cudnn8.1-cuda11.2/8.1.1.33

module list

python  src/recsys23/main.py 
