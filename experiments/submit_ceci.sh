#!/bin/bash
#SBATCH --job-name=enc_eval_<TO_CHANGE>
#SBATCH --time=02:00:00                 # Arrange according to need
#SBATCH --constraint=EPYC               # Request nodes with EPYC CPUs
#SBATCH --output=<TO_CHANGE>.out 
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
 
 module load libffi/3.4.4-GCCcore-13.2.0
 module load bzip2/1.0.8-GCCcore-13.2.0
 python ciphertext-domain/<TO_CHANGE>.py
  

