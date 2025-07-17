import keras as k
from keras import models
from keras import layers

def get_model():
    img_height = 218
    img_width = 152
    img_channels = 3

    inputs = k.Input(shape=(img_height, img_width, img_channels))

    x = layers.Conv2D(20, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(44, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(24, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(48, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(176, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model