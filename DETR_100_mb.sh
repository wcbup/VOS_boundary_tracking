#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J DETR_100_mb
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -u "s232248@dtu.dk"
#BSUB -o log/DETR_100_mb.out
#BSUB -e log/DETR_100_mb.err


module load cuda/11.8

conda init bash
source /work3/s232248/miniconda3/bin/activate
conda activate ras

python DETR_100_mb.py