#!/dls/science/groups/i23/scripts/chris/TFODCourse/tfod/bin/python

import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime

parallel = True
now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tmpdir = "/dls/tmp/vwg85559"

def run():
    print("Using TensorFlow v%s" % tf.__version__)
    
    cwd = os.getcwd()
    data_dir = os.path.join(tmpdir, "goniopin_auto_18022025_binary")
    batch_size = 64
    img_width = 800  # 160 #1292
    img_height = 800  # 250 #964

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

    # random_seed = random.randint(11111111, 99999999)

    # train_datagen = ImageDataGenerator(rescale=1/255.)
    # train_generator = train_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True,
    #     seed=random_seed
    # )

    # class_labels = train_generator.classes

    # indices = np.arange(len(class_labels))
    # train_indices, val_indices, train_labels, val_labels = train_test_split(
    #     indices, class_labels, test_size=0.2, stratify=class_labels, random_state=random_seed, shuffle=True
    # )

    # train_generator.reset()
    # train_generator_subset = train_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True,
    #     seed=random_seed,
    #     subset='training'
    # )
    # train_generator_subset.samples = len(train_indices)
    # train_generator_subset._filepaths = [train_generator._filepaths[i] for i in train_indices]
    # train_generator_subset.classes = [train_labels[i] for i in train_indices]
    # train_generator_subset._targets = np.asarray([train_labels[i] for i in train_indices])

    # val_generator = train_datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=False,
    #     seed=random_seed,
    #     subset='validation'
    # )
    # val_generator.samples = len(val_indices)
    # val_generator._filepaths = [train_generator._filepaths[i] for i in val_indices]
    # val_generator.classes = [val_labels[i] for i in val_indices]
    # val_generator._targets = np.asarray([val_labels[i] for i in val_indices])

    class_names = train_ds.classes
    print("Class names and their corresponding indices:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(192))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        keras.optimizers.Adam(0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", "AUC"],
    )

    model.summary()
    log_dir = "logs/fit/" + now_string
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="batch")

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{now_string}_save_binary_batch{str(batch_size)}.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, verbose=1
        ),
        tensorboard_callback,
    ]

    model.fit(train_ds, callbacks=callbacks, epochs=100, validation_data=val_ds)

    model.save(f"{now_string}_binary_batch{str(batch_size)}.h5")

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    if not parallel:
        run()
    else:
        with strategy.scope():
            run()
