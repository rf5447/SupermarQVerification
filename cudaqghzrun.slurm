#!/bin/bash
#SBATCH --job-name=cudaq_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:05:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rf5447@princeton.edu
#SBATCH --output=cudaqghzrun_%j.out
#SBATCH --error=cudaqghzrun_%j.out


module purge
module load anaconda3/2024.10
module load nvhpc/25.5
module load cudatoolkit/12.6
module load openmpi/nvhpc-25.5/4.1.8

conda activate cudaq-env

export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_btl='^openib'

/usr/bin/time -v python3 cudaqghzrun.py