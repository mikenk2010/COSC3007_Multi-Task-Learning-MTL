#!/usr/bin/env python3
"""
Generate trained model files in both .h5 and .keras formats.
Based EXACTLY on submission_FINAL NOTEBOOK_FINAL_REDUCED.ipynb
"""
# === Exact imports from notebook ===
import os
import random
import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# === Load dataset (exact code from notebook) ===
data = np.load('dataset_dev_3000.npz')
X = data['X']
y = data['y']

# === Train/validation split (exact code from notebook) ===
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y[:, 0],  # Stratify on Target A (10 classes) - like test_clean.ipynb
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# === Prepare data for multi-task training (exact code from notebook) ===
X_train_mtl = X_train[..., None].astype('float32')
X_val_mtl = X_val[..., None].astype('float32')

# Normalize
mean = X_train_mtl.mean()
std = X_train_mtl.std() + 1e-6
X_train_mtl = (X_train_mtl - mean) / std
X_val_mtl = (X_val_mtl - mean) / std

# Extract targets
y_A_train, y_B_train, y_C_train = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_A_val, y_B_val, y_C_val = y_val[:, 0], y_val[:, 1], y_val[:, 2]

# === build_mtl_model (EXACT code from notebook) ===
def build_mtl_model(input_shape=(32, 32, 1)):
    inputs = Input(shape=input_shape)

    # -------- Shared stem (VERY shallow) --------
    x = Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)

    # -------- Target A head (strong) --------
    a = Conv2D(64, 3, padding="same", activation="relu")(x)
    a = MaxPooling2D(2)(a)
    a = Flatten()(a)
    a = Dense(128, activation="relu")(a)
    a = Dropout(0.6)(a)
    output_A = Dense(10, activation="softmax", name="output_A")(a)

    # -------- Target B head (wider, less deep) --------
    b = GlobalAveragePooling2D(name="B_gap")(x)
    b = Dense(64, activation="relu", name="B_dense")(b)
    b = Dropout(0.6, name="B_dropout")(b)
    output_B = Dense(32, activation="softmax", name="output_B")(b)


    # -------- Target C head (light regression) --------
    c = GlobalAveragePooling2D()(x)
    c = Dense(64, activation="relu")(c)
    output_C = Dense(1, activation="linear", name="output_C")(c)

    model = Model(
        inputs=inputs,
        outputs={
            "output_A": output_A,
            "output_B": output_B,
            "output_C": output_C,
        },
    )

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss={
            "output_A": "sparse_categorical_crossentropy",
            "output_B": "sparse_categorical_crossentropy",
            "output_C": "mse",
        },
        loss_weights={
            "output_A": 1.2,  # protect A
            "output_B": 0.7,
            "output_C": 0.4,  # prevent dominance
        },
        metrics={
            "output_A": "accuracy",
            "output_B": "accuracy",
            "output_C": "mae",
        },
    )

    return model


# === Build the model ===
print("Building model from scratch...")
hypothesis_model = build_mtl_model()
hypothesis_model.summary()

# === train() function (EXACT code from notebook) ===
def train():

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ]

    classes = np.unique(y_B_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_B_train,
    )
    class_weight_B = dict(zip(classes, weights))

    sample_weight = {
        "output_A": np.ones(len(y_A_train), dtype=np.float32),
        "output_B": np.array(
            [class_weight_B[y] for y in y_B_train],
            dtype=np.float32,
        ),
        "output_C": np.ones(len(y_C_train), dtype=np.float32),
    }

    history = hypothesis_model.fit(
        X_train_mtl,
        {
            "output_A": y_A_train,
            "output_B": y_B_train,
            "output_C": y_C_train,
        },
        sample_weight=sample_weight,
        validation_data=(
            X_val_mtl,
            {
                "output_A": y_A_val,
                "output_B": y_B_val,
                "output_C": y_C_val,
            },
        ),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    return history

# === Train the model ===
history_mtl = train()

# === Evaluate (exact code from notebook) ===
print("\n" + "=" * 60)
print("HYPOTHESIS-DRIVEN MTL MODEL EVALUATION")
print("=" * 60)

# Get predictions (DICT)
preds = hypothesis_model.predict(X_val_mtl, verbose=0)

y_pred_A = preds["output_A"]
y_pred_B = preds["output_B"]
y_pred_C = preds["output_C"]

# Convert predictions to class labels
y_pred_A_labels = np.argmax(y_pred_A, axis=1)
y_pred_B_labels = np.argmax(y_pred_B, axis=1)

# Calculate metrics
acc_A = np.mean(y_pred_A_labels == y_A_val)
acc_B = np.mean(y_pred_B_labels == y_B_val)
mae_C = np.mean(np.abs(y_pred_C.squeeze() - y_C_val))

print("\nTarget A (Global Shape/Geometry):")
print(f"  Accuracy: {acc_A*100:.2f}%")
print(f"  Random baseline: {1/10*100:.2f}%")

print("\nTarget B (Orientation/Fine Structure):")
print(f"  Accuracy: {acc_B*100:.2f}%")
print(f"  Random baseline: {1/32*100:.2f}%")

print("\nTarget C (Intensity/Amplitude):")
print(f"  MAE: {mae_C:.4f}")
print(f"  Range: [{y_C_val.min():.4f}, {y_C_val.max():.4f}]")

# === Save model files ===
MODEL_H5 = "model_s3715228_s3343711_s4139514.h5"
MODEL_KERAS = "model_s3715228_s3343711_s4139514.keras"

print(f"\nSaving {MODEL_H5}...")
hypothesis_model.save(MODEL_H5)
print(f"  Saved: {MODEL_H5} ({os.path.getsize(MODEL_H5) / (1024*1024):.2f} MB)")

print(f"\nSaving {MODEL_KERAS}...")
hypothesis_model.save(MODEL_KERAS)
print(f"  Saved: {MODEL_KERAS} ({os.path.getsize(MODEL_KERAS) / (1024*1024):.2f} MB)")

print("\n" + "=" * 60)
print("MODEL FILES GENERATED SUCCESSFULLY")
print("=" * 60)
