tfimage=/dls_sw/apps/tensorflow/singularity/tensorflow_2.8.2-gpu-jupyter.sif

singularity exec --nv --home /dls/tmp/vwg85559 $tfimage python ./img_classification_binary_working.py