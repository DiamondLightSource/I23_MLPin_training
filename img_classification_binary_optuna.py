import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
import tensorflow as tf
import tensorflow.keras.backend as K

def build_model(config):
    model = models.Sequential()
    
    hp_units_1 = config["units_1"]
    model.add(layers.Conv2D(hp_units_1, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(hp_units_1, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    hp_units_2 = config["units_2"]
    model.add(layers.Conv2D(hp_units_2, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(hp_units_2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    hp_dense_units = config["dense_units"]
    model.add(layers.Dense(hp_dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def objective(trial):
    # Define hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    units_1 = trial.suggest_int('units_1', 32, 128, step=32)
    units_2 = trial.suggest_int('units_2', 32, 128, step=32)
    dense_units = trial.suggest_int('dense_units', 32, 128, step=32)

    config = {
        "units_1": units_1,
        "units_2": units_2,
        "dense_units": dense_units
    }

    # Build and compile model
    model = build_model(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2),
        TFKerasPruningCallback(trial, 'val_loss')
    ]

    # Train the model
    history = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0
    )

    # Clear Keras session to free memory
    K.clear_session()

    # Return the validation accuracy
    return history.history['val_accuracy'][-1]

if __name__ == "__main__":
    import argparse
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args = parser.parse_args()

    data_dir = os.path.join(cwd, "goniopin_auto_19022025_binary")
    # Create the datasets
    datagen = ImageDataGenerator(validation_split=0.2)
    train_ds = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    val_ds = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=4)  # Adjust n_jobs based on your GPU availability

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))