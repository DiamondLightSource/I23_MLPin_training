import os
import random
import datetime
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
import tensorflow.keras.backend as K
import keras_tuner as kt

parallel = False
now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tmpdir = "/dls/tmp/vwg85559"

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def run():
    print("Using TensorFlow v%s" % tf.__version__)

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "goniopin_auto_19022025_binary")
    batch_size = 4
    img_height = 800
    img_width = 800
    seed = random.randint(11111111, 99999999)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="binary",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="binary",
    )

    def model_builder(hp):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
        model.add(layers.Rescaling(1.0 / 255))

        hp_units_1 = hp.Int("units_1", min_value=4, max_value=64, step=4)
        model.add(layers.Conv2D(hp_units_1, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(hp_units_1, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        hp_units_2 = hp.Int("units_2", min_value=8, max_value=128, step=4)
        model.add(layers.Conv2D(hp_units_2, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(hp_units_2, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        hp_dense_units = hp.Int("dense_units", min_value=16, max_value=96, step=16)
        model.add(layers.Dense(hp_dense_units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))
        hp_dense_units_2 = hp.Int("dense_units_1", min_value=8, max_value=48, step=8)
        model.add(layers.Dense(hp_dense_units_2))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(1, activation="sigmoid"))

        hp_learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall", "AUC"],
        )

        return model

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=15,
        factor=3,
        directory="my_tuner_dir",
        project_name="binary_classification_tuning",
    )

    tuner.search(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
    )
    K.clear_session()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameters search is complete. The best hyperparameters are:
    Conv units: {best_hps.get("units_1")},
    Conv units 2: {best_hps.get("units_2")},
    Dense units: {best_hps.get("dense_units")},
    Dense units 2: {best_hps.get("dense_units_2")},
    Learning rate: {best_hps.get("learning_rate")}.
    """)

    model = model_builder(best_hps)

    model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                f"{now_string}_best_model.h5", save_best_only=True
            ),
        ],
    )

    loss, accuracy, precision, recall, auc = model.evaluate(val_ds)
    print(
        f"Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC: {auc}"
    )

    model.save(f"{now_string}_tuned_model.h5")


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    if not parallel:
        run()
    else:
        with strategy.scope():
            run()
