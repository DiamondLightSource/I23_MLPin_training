#!/dls/science/groups/i23/scripts/chris/TFODCourse/tfod/bin/python

import os
import random
from gc import callbacks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

parallel = False


def run():
    print("Using TensorFlow v%s" % tf.__version__)

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "gripperstatus_auto_25052023")
    batch_size = 32
    img_height = 300  # 250 #964
    img_width = 160  # 160 #1292
    seed = random.randint(11111111,99999999)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical",
    )

    data_augmentation = Sequential(
        [
            keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.2, fill_mode="nearest"
            ),
            keras.layers.RandomContrast(factor=0.2),
            keras.layers.RandomBrightness(factor=0.2),
            keras.layers.RandomRotation(0.001, fill_mode="nearest"),
        ]
    )

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height,img_width,3)))
    model.add(data_augmentation)
    model.add(layers.Rescaling(1.0 / 255))
    
    model.add(layers.Conv2D(32, 3, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(192))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(keras.optimizers.Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        ),
    ]

    model.fit(train_ds, callbacks=callbacks, epochs=5, validation_data=val_ds)

    model.save("gripper.h5")

strategy = tf.distribute.MirroredStrategy()
if not parallel:
    run()
else:
    with strategy.scope():
        run()