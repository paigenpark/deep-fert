import tensorflow as tf
import numpy as np
import os as os
tfkl = tf.keras.layers

# --- Year feature normalization ---------------------------------------------
# Affine (min-max) transform anchored to YEAR_MIN..YEAR_MAX. Centralized here so
# training (get_data) and forecasting apply the exact same mapping and can't
# drift. Plain arithmetic, so it works on Python scalars, numpy arrays, and TF
# tensors alike. Years outside the anchor range map outside [0, 1] (expected --
# e.g. forecast years extrapolate beyond it), which is fine for a linear input.
YEAR_MIN = 1950
YEAR_MAX = 2015


def normalize_year(year):
    return (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)


# get and prepare data
def get_data(index, data, max_val, mode, changeratetolog=False):
    if mode == "train":
        # Randomly selects index from training data between 0 and the max index in train
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32) 
        entry = data[rand_index, :]
    elif mode == "not_random":
        # Selects specified index from test data 
        entry = data[index, :]
    else:  # Assuming mode="test" or any other value
        # For any other value of mode, randomly selects index from test
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32)
        entry = data[rand_index, :]

    geography, year, age, rate = entry[0], entry[1], entry[2], entry[3]

    # Normalization or preparation
    year = normalize_year(year)
    age = tf.cast(age, tf.int32)
    geography = tf.cast(geography, tf.int32)
    if changeratetolog:
        epsilon = 1e-05 # min rate in training data
        rate = tf.math.log(tf.maximum(rate, epsilon))

    # Reshape each element to scalar
    features = (tf.reshape(year, [1]), tf.reshape(age, [1]), 
                tf.reshape(geography, [1]))
    rate = tf.reshape(rate, [1])
    return features, rate

    
def prep_data(data, mode, changeratetolog=False):
    
    data = tf.convert_to_tensor(data)
    data = tf.cast(data, tf.float32)
    max_val = data.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices(np.arange(3000))

    if mode == "train":
        dataset = dataset.repeat()
    
    else:
        dataset = dataset.repeat(120)
    
    dataset = dataset.map(
        lambda x: get_data(x, data, max_val=max_val, mode=mode, changeratetolog=changeratetolog), 
                          num_parallel_calls=4)

    # Batch the dataset for efficient predictions 
    # Each batch consists of 2 parts - batch of features and batch of targets
    dataset = dataset.batch(256)

    # Prefetch to improve performance
    final_data = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_data

# create DL model
#
# Hyperparameters (units, n_layers, dropout, embedding dims, learning_rate) are
# exposed as arguments so a tuner can search over them. The DEFAULTS below
# reproduce the original hardcoded architecture exactly, so existing callers that
# pass only geo_dim get identical models to before.
DEFAULT_HPARAMS = {
    "units": 64,
    "n_layers": 3,
    "dropout": 0.1,
    "age_embed_dim": 5,
    "geo_embed_dim": 5,
    "learning_rate": 1e-3,  # Adam's default, matching optimizer='adam'
}


def build_model(geo_dim, lograte=False, units=64, n_layers=3, dropout=0.1,
                age_embed_dim=5, geo_embed_dim=5, learning_rate=1e-3):
    """Build the ASFR model. `lograte=False` -> sigmoid output (raw rates);
    `lograte=True` -> linear output (log rates)."""
    # defining inputs
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age = tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')

    # defining embedding layers
    age_embed = tfkl.Embedding(input_dim=55, output_dim=age_embed_dim, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=geo_embed_dim, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    # create feature vector that concatenates all inputs
    x = tfkl.Concatenate()([year, age_embed, geography_embed])
    x1 = x

    # setting up middle layers
    for _ in range(n_layers):
        x = tfkl.Dense(units, activation='relu')(x)
        x = tfkl.LayerNormalization()(x)
        x = tfkl.Dropout(dropout)(x)

    # setting up output layer (residual concat from inputs)
    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(units, activation='relu')(x)
    x = tfkl.LayerNormalization()(x)
    x = tfkl.Dropout(dropout)(x)

    output_activation = None if lograte else 'sigmoid'
    x = tfkl.Dense(1, activation=output_activation, name='final')(x)

    # creating and compiling the model
    model = tf.keras.Model(inputs=[year, age, geography], outputs=[x])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model


# Backward-compatible wrappers so existing callers keep working unchanged.
def create_model(geo_dim, **hparams):
    return build_model(geo_dim, lograte=False, **hparams)


def create_log_model(geo_dim, **hparams):
    return build_model(geo_dim, lograte=True, **hparams)


# run DL model
def run_deep_model(dataset_train, dataset_test, geo_dim, epochs, steps_per_epoch, lograte=False):
    if lograte:
        model = create_log_model(geo_dim)
    else:
        model = create_model(geo_dim)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,            # Wait 10 epochs before giving up
        verbose=1,
        mode="auto",
        restore_best_weights=True # Crucial: reverts model to its best state
        )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.25, 
        patience=3, 
        verbose=1, 
        min_delta=1e-8
        )
    
    history = model.fit(dataset_train, validation_data=dataset_test, validation_steps=25, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, verbose=2, callbacks=[early_stopping, reduce_lr])

    val_loss = min(history.history['val_loss'])

    tf.keras.backend.clear_session()

    return model, val_loss


# run DL model with leak-free two-phase "refit" schedule
#
# Phase 1: fit on a temporal-holdout subset (years <= JOY-GAP) and early-stop on
#          a held-out validation set (years JOY-GAP+1..JOY) to pick the best epoch.
# Phase 2: refit a fresh model on the FULL training set (all years <= JOY) for that
#          many epochs, with NO validation peek. This keeps model selection honest
#          (never touches post-JOY graded data) while recovering the recent years
#          that a plain temporal holdout would drop from training.
def run_deep_model_refit(dataset_train_sub, dataset_val, dataset_train_full, geo_dim,
                         epochs, steps_per_epoch_sub, steps_per_epoch_full, lograte=False,
                         hparams=None):
    hparams = hparams or {}
    build = lambda: build_model(geo_dim, lograte=lograte, **hparams)

    # --- Phase 1: find best epoch via temporal-holdout validation ---
    model = build()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True
        )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.25,
        patience=3,
        verbose=1,
        min_delta=1e-8
        )

    history = model.fit(dataset_train_sub, validation_data=dataset_val, validation_steps=25,
                        steps_per_epoch=steps_per_epoch_sub, epochs=epochs, verbose=2,
                        callbacks=[early_stopping, reduce_lr])

    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    best_val_loss = min(history.history['val_loss'])
    tf.keras.backend.clear_session()

    # --- Phase 2: refit on full <=JOY data for best_epoch epochs, no val peek ---
    model = build()

    # No validation set in phase 2, so schedule LR off the training loss instead.
    reduce_lr_full = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.25,
        patience=3,
        verbose=1,
        min_delta=1e-8
        )

    model.fit(dataset_train_full, steps_per_epoch=steps_per_epoch_full, epochs=best_epoch,
              verbose=2, callbacks=[reduce_lr_full])

    tf.keras.backend.clear_session()

    return model, best_epoch, best_val_loss

