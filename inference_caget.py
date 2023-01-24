#!/dls/science/groups/i23/scripts/chris/TFODCourse/tfod/bin/python

import tensorflow as tf
from tensorflow import keras
import cv2
import os
from io import BytesIO
import urllib
from time import sleep
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
import ca
import pv

model = tf.keras.models.load_model("final.h5")
model.summary()
now = datetime.now()
today = now.strftime("%d%m%Y")
cwd = os.getcwd()
path = os.path.join(cwd, f"goniopin_infer_{today}")
pinon = os.path.join(path, "pinon")
pinoff = os.path.join(path, "pinoff")
unsure = os.path.join(path, "unsure")


def run():
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    for decision in (pinon, pinoff, unsure):
        if os.path.exists(decision):
            pass
        else:
            os.mkdir(decision)


def urltoimage(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[400:700, 610:770]
    cv2.imwrite("tmp.jpg", image)
    _, buffer = cv2.imencode(".jpg", image)
    io_buf = BytesIO(buffer)
    return io_buf


def infer():
    now = datetime.now()
    this_second = now.strftime("%m%d%Y%H%M%S")
    stream = urltoimage("http://bl23i-di-serv-02.diamond.ac.uk:8080/ECAM6.mjpg.jpg")
    img_in = keras.preprocessing.image.load_img((stream), target_size=(300, 160))
    plt.imshow(img_in)
    img_array = keras.preprocessing.image.img_to_array(img_in)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    if score > 0.95:
        print(f"{100 * score}% sure pin is ON")
        plt.savefig(os.path.join(pinon, this_second + "_" + str(int(100 * score))))
    elif score < 0.05:
        print(f"{100 * (1 - score)}% sure pin is OFF")
        plt.savefig(
            os.path.join(pinoff, this_second + "_" + str(int(100 * (1 - score))))
        )
    else:
        print("No idea.")
        plt.savefig(os.path.join(unsure, this_second))    

if __name__ == "__main__":
    run()
    while True:
        if np.round(float(ca.caget(pv.grip_x_rbv)), 1) != 0.4:
            infer()
            sleep(1)
        else:
            sleep(1)
