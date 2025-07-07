import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(
    model_path="/dls/science/groups/i23/scripts/chris/I23_MLPin_training/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model/quant_detect.tflite"
)
# interpreter = tf.lite.Interpreter(model_path="/scratch/docker/coral/quant_detect_edgetpu.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture("http://bl23i-di-serv-02.diamond.ac.uk:8080/ECAM6.mjpg.mjpg")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
frame = cv2.resize(frame, [320, 320])
image_np = np.array(frame)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
interpreter.set_tensor(input_details[0]["index"], input_tensor)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])
print(output_data)

cv2.imshow("Image", frame)
cv2.waitKey(5000)
