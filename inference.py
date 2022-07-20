import tensorflow as tf
from tensorflow import keras
import img_classification
import cv2

model = tf.keras.models.load_model("save_at_49.h5")
model.summary()

#img_in = keras.preprocessing.image.load_img("", target_size=img_classification.image_size)

