#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J cotracker_win
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -u "s232248@dtu.dk"
#BSUB -o log/cotracker_win.out
#BSUB -e log/cotracker_win.err


module load cuda/11.8

conda init bash
source /work3/s232248/miniconda3/bin/activate
conda activate rd

python cotracker_win.py