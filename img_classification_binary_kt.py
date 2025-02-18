import os
import random
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

parallel = True
now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tmpdir = "/dls/tmp/vwg85559"

def run():
    print("Using TensorFlow v%s" % tf.__version__)
    
    cwd = os.getcwd()
    data_dir = os.path.join(tmpdir, "goniopin_auto_17022025_binary")
    batch_size = 64
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

    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     horizontal_flip=True,
    #     brightness_range=[0.8, 1.2]
    # )

    # val_datagen = ImageDataGenerator(rescale=1./255)

    # train_generator = train_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True,
    #     seed=seed
    # )

    # class_labels = train_generator.classes
    # indices = np.arange(len(class_labels))

    # train_indices, val_indices, train_labels, val_labels = train_test_split(
    #     indices, class_labels, test_size=0.2, stratify=class_labels, random_state=seed, shuffle=True
    # )

    # train_generator_subset = train_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True,
    #     seed=seed
    # )
    # train_generator_subset.samples = len(train_indices)
    # train_generator_subset._filepaths = [train_generator._filepaths[i] for i in train_indices]
    # train_generator_subset.classes = [train_labels[i] for i in train_indices]
    # train_generator_subset._targets = np.asarray([train_labels[i] for i in train_indices])

    # val_generator = val_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True,
    #     seed=seed
    # )
    # val_generator.samples = len(val_indices)
    # val_generator._filepaths = [train_generator._filepaths[i] for i in val_indices]
    # val_generator.classes = [val_labels[i] for i in val_indices]
    # val_generator._targets = np.asarray([val_labels[i] for i in val_indices])

    def model_builder(hp):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
        model.add(layers.Rescaling(1./255))

        hp_units_1 = hp.Int('units_1', min_value=4, max_value=64, step=4)
        model.add(layers.Conv2D(hp_units_1, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(hp_units_1, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        hp_units_2 = hp.Int('units_2', min_value=4, max_value=64, step=4)
        model.add(layers.Conv2D(hp_units_2, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(hp_units_2, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        hp_dense_units = hp.Int('dense_units', min_value=16, max_value=256, step=16)
        model.add(layers.Dense(hp_dense_units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation="sigmoid"))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-2, 1e-3])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall", "AUC"]
        )

        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=50,
        directory='my_tuner_dir',
        project_name='binary_classification_tuning'
    )

    tuner.search(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameters search is complete. The best hyperparameters are:
    Conv units: {best_hps.get('units_1')},
    Conv units 2: {best_hps.get('units_2')},
    Dense units: {best_hps.get('dense_units')},
    Learning rate: {best_hps.get('learning_rate')}.
    """)

    model = model_builder(best_hps)

    model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.ModelCheckpoint(f"{now_string}_best_model.h5", save_best_only=True)
        ]
    )

    loss, accuracy, precision, recall, auc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC: {auc}")

    model.save(f"{now_string}_tuned_model.h5")

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    if not parallel:
        run()
    else:
        with strategy.scope():
            run()
