#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J cotracker_light_base
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -u "s232248@dtu.dk"
#BSUB -o log/cotracker_light_base.out
#BSUB -e log/cotracker_light_base.err


module load cuda/11.8

conda init bash
source /work3/s232248/miniconda3/bin/activate
conda activate rd

python cotracker_light_base.py