from gc import callbacks
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

print("Using TensorFlow v%s" % tf.__version__)
acc_str = "accuracy" if tf.__version__[:2] == "2." else "acc"

#data_dir = pathlib.Path("C:/Users/ULTMT/Documents/code/TFOD/I23_MLPin_training/goniopin/cropped")
cwd = os.getcwd()
data_dir = os.path.join(cwd, "cropped")
batch_size = 16
img_height = 250
img_width = 250
image_size = (img_height, img_width)
seed = 28273492


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

plt.show()

normalization_layer = keras.layers.Rescaling(
    1.0 / 255
)

data_augmentation = Sequential(
    [
        keras.layers.RandomFlip(input_shape=(img_height, img_width, 3)),
        keras.layers.RandomRotation(5),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


# model = Sequential()
# model.add(data_augmentation)
# model.add(normalization_layer)
# model.add(Flatten(input_shape=(img_width, img_height)))
# #model.add(Dense(128, activation="relu"))
# #model.add(Dropout(0.5))
# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.5))
# #model.add(Dense(32, activation="relu"))
# model.add(Dense(2))

# train_ds = train_ds.prefetch(buffer_size=32)
# val_ds = val_ds.prefetch(buffer_size=32)

# model.build()
# model.summary()

# model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])


# cp_callback = keras.callbacks.ModelCheckpoint(filepath="checkpoints/", verbose=1, save_weights_only=True, save_freq=64*batch_size)
# epochs = 100
# training = model.fit(
#     train_ds, epochs=epochs, batch_size=batch_size, validation_data=(val_ds), callbacks=cp_callback
# )

# # plot accuracy
# plt.figure(dpi=100, figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(training.history[acc_str], label="Accuracy on training data")
# plt.plot(training.history["val_" + acc_str], label="Accuracy on test data")
# plt.legend()
# plt.title("Accuracy")

# # plot loss
# plt.subplot(1, 2, 2)
# plt.plot(training.history["loss"], label="Loss on training data")
# plt.plot(training.history["val_loss"], label="Loss on test data")
# plt.legend()
# plt.title("Loss")
# plt.show()