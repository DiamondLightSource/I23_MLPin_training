import tensorflow as tf
from tensorflow import keras
import cv2
from io import BytesIO
import urllib
from time import sleep
import numpy as np

# Load Keras model
model = tf.keras.models.load_model("categorical.h5")

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('categorical.tflite', 'wb') as f:
    f.write(tflite_model)

