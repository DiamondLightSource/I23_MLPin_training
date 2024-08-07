#!/bin/bash
#SBATCH --job-name=i23pinml
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --partition=cs05r
#SBATCH --time=23:59:59


tfimage=/dls_sw/apps/tensorflow/singularity/tensorflow_2.8.2-gpu-jupyter.sif
echo Training
singularity exec --nv --home $PWD $tfimage python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config --num_train_steps=2000
echo Evaluating
singularity exec --nv --home $PWD $tfimage python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
echo Freezing
singularity exec --nv --home $PWD $tfimage python Tensorflow/models/research/object_detection/exporter_main_v2.py  --input_type=image_tensor --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --output_directory=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/export
echo Converting to TFJS
singularity exec --nv --home $PWD $tfimage tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/export/saved_model Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tfjsexport
echo Converting to TFLite
singularity exec --nv --home $PWD $tfimage python Tensorflow/models/research/object_detection/export_tflite_graph_tf2.py  --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --output_directory=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tfliteexport
singularity exec --nv --home $PWD $tfimage tflite_convert --saved_model_dir=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tfliteexport/saved_model --output_file=Tensorflow/workspace/models/my_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tfliteexport/saved_model/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops
