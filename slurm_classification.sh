#!/bin/bash
#SBATCH --job-name=i23pinml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:1
#SBATCH --partition=cs05r
#SBATCH --time=23:59:59
#export TF_GPU_ALLOCATOR=cuda_malloc_async
tfimage=/dls_sw/apps/tensorflow/singularity/tensorflow_2.8.2-gpu-jupyter.sif

singularity exec --nv --home $PWD $tfimage python ./img_classification_binary_kt.py
