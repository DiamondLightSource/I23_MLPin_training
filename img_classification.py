import os
import numpy as np
from gc import callbacks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt

parallel = True

def run():
    print("Using TensorFlow v%s" % tf.__version__)
    acc_str = "accuracy" if tf.__version__[:2] == "2." else "acc"

    # data_dir = pathlib.Path("C:/Users/ULTMT/Documents/code/TFOD/I23_MLPin_training/goniopin/cropped")
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "goniopin_auto_12012023")
    batch_size = 32
    img_height = 300  # 250 #964
    img_width = 160  # 160 #1292
    image_size = (img_height, img_width)
    seed = np.randint(11111111,99999999)

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

    # wont work with singularity as it tries to plot image
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig(os.path.join(cwd, "un-augmented.png"))

    data_augmentation = Sequential(
        [
            keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.2, fill_mode="nearest"
            ),
            keras.layers.RandomContrast(factor=0.2),
            keras.layers.RandomBrightness(factor=0.3),
        ]
    )

    plt.figure(figsize=(20, 20))
    for images, _ in train_ds.take(1):
        for i in range(25):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(augmented_images[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.savefig(os.path.join(cwd, "augmented.png"))

    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = layers.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

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

    epochs = 100

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=15, restore_best_weights=True
        ),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "mae", "categorical_accuracy"],
    )

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.save("final.h5")


strategy = tf.distribute.MirroredStrategy()
if not parallel:
    run()
else:
    with strategy.scope():
        run()