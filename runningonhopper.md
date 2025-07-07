To run on a specific partition with 4 GPUs and 5 tasks per node, you can use the following command:

srun --nodes=1 --partition=cs05r --ntasks-per-node=5 --gres=gpu:4 --pty bash

Then load the singularity image:

tfimage=/dls_sw/apps/tensorflow/singularity/tensorflow_2.8.2-gpu-jupyter.sif

Then run the python script:

singularity exec --nv --home $PWD $tfimage python ./img_classification_binary_kt.py