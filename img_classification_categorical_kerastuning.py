#!/dls/science/groups/i23/scripts/chris/TFODCourse/tfod/bin/python
import os
import random
from gc import callbacks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import keras_tuner as kt

parallel = True
tf.get_logger().setLevel("ERROR")


def run():
    print("Using TensorFlow v%s" % tf.__version__)

    # data_dir = pathlib.Path("C:/Users/ULTMT/Documents/code/TFOD/I23_MLPin_training/goniopin/cropped")
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "goniopin_auto_0808023")
    img_height = 300  # 250 #964
    img_width = 160  # 160 #1292
    seed = random.randint(11111111, 99999999)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        label_mode="categorical",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        label_mode="categorical",
    )

    # Augment data
    data_augmentation = Sequential(
        [
            keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.2, fill_mode="nearest"
            ),
            # keras.layers.RandomContrast(factor=0.2),
            keras.layers.RandomBrightness(factor=0.2),
            keras.layers.RandomRotation(0.02, fill_mode="nearest"),
        ]
    )

    def model_builder(hp):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
        model.add(data_augmentation)
        model.add(layers.Rescaling(1.0 / 255))

        model.add(layers.Conv2D(32, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        model.add(layers.Dense(units=hp_units))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4, activation="softmax"))

        hp_learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-5)
        model.compile(
            keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", "mae"],
        )

        return model

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=30,
        factor=3,
        hyperband_iterations=3,
        directory="tuning",
        project_name="img_classification_categorical",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    tuner.search(train_ds, callbacks=callbacks, epochs=50, validation_data=val_ds)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """
    )

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, epochs=50, validation_split=0.2, callbacks=callbacks)

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(train_ds, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(val_ds)
    print("[test loss, test accuracy]:", eval_result)

    hypermodel.save("hyper.h5")


strategy = tf.distribute.MirroredStrategy()
if not parallel:
    run()
else:
    with strategy.scope():
        run()
