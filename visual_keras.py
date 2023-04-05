#!/dls/science/groups/i23/scripts/chris/TFODCourse/tfod/bin/python

import visualkeras as vk
import tensorflow as tf
from PIL import ImageFont
from collections import defaultdict
from keras import layers

color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#00f5d4'
color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
color_map[layers.Dropout]['fill'] = '#03045e'
color_map[layers.Dense]['fill'] = '#fb5607'
color_map[layers.Flatten]['fill'] = '#ffbe0b'

font = ImageFont.truetype("arial.ttf", 12)
model = tf.keras.models.load_model("categorical.h5")
model.summary()

v = vk.layered_view(model=model, legend=True, font=font, color_map=color_map)
v.show()

k = vk.layered_view(model=model, legend=True, font=font, draw_volume=False, color_map=color_map)
k.show()