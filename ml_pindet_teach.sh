tfimage=/dls_sw/apps/tensorflow/singularity/tensorflow_2.8.2-gpu-jupyter.sif

#singularity exec --nv --home $PWD $tfimage python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8 --pipeline_config_path=Tensorflow/workspace/models/my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config --num_train_steps=2000

singularity exec --nv --home $PWD $tfimage python img_classification.py