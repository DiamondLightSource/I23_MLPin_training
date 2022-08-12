import numpy as np
import tensorflow as tf


def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 320, 320, 3)
        yield [data.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model(
    "/dls/science/groups/i23/scripts/chris/I23_MLPin_training/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model"
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()
with open(
    "/dls/science/groups/i23/scripts/chris/I23_MLPin_training/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model/quant_detect.tflite",
    "wb",
) as quant_out:
    quant_out.write(tflite_quant_model)

interpreter = tf.lite.Interpreter(
    model_path="/dls/science/groups/i23/scripts/chris/I23_MLPin_training/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model/quant_detect.tflite"
)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
# print(input_details)
output_details = interpreter.get_output_details()
# print(output_details)

# Test the model on random input data.
input_shape = input_details[0]["shape"]
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# print(input_data)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]["index"])
print(output_data)
