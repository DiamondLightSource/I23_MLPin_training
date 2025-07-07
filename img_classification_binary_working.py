import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
import matplotlib.pyplot as plt
import cv2
cv2.setNumThreads(0)

parallel = True
now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tmpdir = "/dls/tmp/vwg85559"
data_dir_name = "goniopin_auto_24062025_binary"
batch_size = 16
resume_epoch = 47
cont = True # continue from saved epoch
checkpoint_path = f"checkpoints/20250625-161225_epoch47_binary_batch16.h5"


os.makedirs("checkpoints", exist_ok=True)

def get_first_image_size(data_dir):
    pinon_dir = os.path.join(data_dir, "pinon")
    files = [f for f in os.listdir(pinon_dir) if os.path.isfile(os.path.join(pinon_dir, f))]
    if not files:
        raise FileNotFoundError("No files found in pinon directory")
    first_file = os.path.join(pinon_dir, files[0])
    img_bytes = tf.io.read_file(first_file)
    img = tf.image.decode_image(img_bytes)
    height = img.shape[0]
    width = img.shape[1]
    return height, width

def run():
    print("Using TensorFlow v%s" % tf.__version__)

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, data_dir_name)
    img_height, img_width = get_first_image_size(data_dir)

    # img_width = 1182  # 160 #1292
    # img_height = 854  # 250 #964

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
    train_ds = train_ds.prefetch(buffer_size=2)

    # os.makedirs("sample_images", exist_ok=True)
    # for images, labels in train_ds.take(1):
    #     for i in range(min(5, len(images))):
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(f"Label: {labels[i].numpy()}")
    #         plt.axis('off')
    #         plt.savefig(f"sample_images/sample_{i}_label_{int(labels[i].numpy())}.png")
    #         plt.close()

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = val_ds.prefetch(buffer_size=2)

    if cont is False:
        initial_epoch = 1
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
        model.add(layers.Rescaling(1.0 / 255))

        model.add(layers.Conv2D(128, 3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
    #    model.add(layers.GlobalAveragePooling2D())

        # model.add(layers.Dense(144, activation="silu"))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.5))


        model.add(layers.Dense(128, activation="silu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        initial_epoch = resume_epoch
        model = keras.models.load_model(checkpoint_path)
        print(f"Loaded model: {checkpoint_path}")

    model.compile(
        keras.optimizers.Adam(0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", "AUC"],
    )

    model.summary()
    log_dir = "logs/fit/" + now_string
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )

    callbacks = [
	keras.callbacks.ModelCheckpoint(
	    filepath=f"checkpoints/{now_string}_epoch{{epoch:02d}}_binary_batch{batch_size}.h5",
	    save_best_only=False,
	    save_weights_only=False,
	    verbose=1
	),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, verbose=1
        ),
        tensorboard_callback,
    ]

    model.fit(train_ds, callbacks=callbacks, epochs=100, validation_data=val_ds, initial_epoch=initial_epoch)

    model.save(f"{now_string}_binary_batch{str(batch_size)}.h5")


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    if not parallel:
        run()
    else:
        with strategy.scope():
            run()
