import tensorflow as tf
from tensorflow import keras
import cv2
from io import BytesIO
import urllib
from time import sleep
import numpy as np
model = tf.keras.models.load_model("final.h5")
model.summary()

def urltoimage(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[400:750, 610:770]
    cv2.imwrite("tmp.jpg", image)
    _, buffer = cv2.imencode(".jpg", image)
    io_buf = BytesIO(buffer)
    return io_buf

def infer():
    #with urllib.request.urlopen("http://bl23i-di-serv-02.diamond.ac.uk:8080/ECAM6.mjpg.jpg") as url:
    stream = urltoimage("http://bl23i-di-serv-02.diamond.ac.uk:8080/ECAM6.mjpg.jpg")
    img_in = keras.preprocessing.image.load_img((stream), target_size=(250, 160))
    img_array = keras.preprocessing.image.img_to_array(img_in)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    print(score)

if __name__ == "__main__":
    while True:
        infer()
        sleep(2)