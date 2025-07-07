import os
import random
import datetime
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback


def train_model(config, checkpoint_dir=None):
    # Load Data
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "goniopin_auto_19022025_binary")
    batch_size = 16
    img_height = 800
    img_width = 800
    seed = random.randint(11111111, 99999999)

    train_dir = "dls/tmp/vwg85559/ray/session_2025-02-24_16-00-24_585448_1619371/artifacts/2025-02-24_16-00-32/binary_classification_tuning/working_dirs/train_model_79fe9_00031_31_dense_units=64,dropout=False,learning_rate=0.0010,units_1=48,units_2=36_2025-02-24_16-00-33/goniopin_auto_19022025_binary"
    
    # Ensure the directory exists
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
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

    # Define Model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    hp_units_1 = config["units_1"]
    model.add(layers.Conv2D(hp_units_1, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(hp_units_1, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if config["dropout"]:
        model.add(layers.Dropout(0.25))

    hp_units_2 = config["units_2"]
    model.add(layers.Conv2D(hp_units_2, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(hp_units_2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if config["dropout"]:
        model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    hp_dense_units = config["dense_units"]
    model.add(layers.Dense(hp_dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", "AUC"],
    )

    # Define Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2),
        TuneReportCallback(
            {"accuracy": "val_accuracy", "loss": "val_loss"}, on="validation_end"
        ),
    ]

    # Train the Model
    model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,  # Set to 1 for more detailed logs
    )

    # Clear Keras Session to free memory
    K.clear_session()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args = parser.parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define Hyperparameter Search Space
    search_space = {
        "units_1": tune.choice(
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        ),
        "units_2": tune.choice(
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        ),
        "dense_units": tune.choice([16, 32, 48, 64, 80, 96, 112, 128, 144, 160]),
        "dropout": tune.choice([True, False]),
        "learning_rate": tune.choice([1e-4, 1e-3]),
    }

    if args.smoke_test:
        num_samples = 2
        max_num_epochs = 2
        gpus_per_trial = 0
    else:
        num_samples = 100  # Adjust based on your computational budget
        max_num_epochs = 15
        gpus_per_trial = 1

    # Define Scheduler
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=3,
    )

    # Define Ray Tune Run Configuration
    analysis = tune.run(
        train_model,
        resources_per_trial={"gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="file:///dls/science/groups/i23/scripts/chris/I23_MLPin_training/ray_results",  # Directory to save results
        name="binary_classification_tuning",
        verbose=1,
    )

    # Retrieve the Best Trial
    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    best_hps = best_trial.config

    print(f"""
    The hyperparameter search is complete. The best hyperparameters are:
    Conv units 1: {best_hps["units_1"]},
    Conv units 2: {best_hps["units_2"]},
    Dense units: {best_hps["dense_units"]},
    Learning rate: {best_hps["learning_rate"]},
    Dropout: {best_hps["dropout"]}.
    """)

    # Proceed to Train the Final Model with Best Hyperparameters
    # Ensure that you re-initialize the environment (reload data, etc.) as needed
    # This part remains similar to your original script
    # ...

    # Stop Ray
    ray.shutdown()
