#!/usr/bin/env python3
"""Debug script to understand why the model has low accuracy."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

print("TensorFlow version:", tf.__version__)

# Load the trained model
try:
    print("\n1. Loading model...")
    model = keras.models.load_model(
        'model_s3715228_s3343711_s4139514_seed42.h5',
        compile=False
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Print model architecture
print("\n2. Model Architecture:")
print(f"Inputs: {model.inputs}")
print(f"Outputs: {model.outputs}")
print(f"Total layers: {len(model.layers)}")

# Check for Lambda layer with stop_gradient
print("\n3. Checking Lambda layer:")
for layer in model.layers:
    if isinstance(layer, keras.layers.Lambda):
        print(f"Found Lambda layer: {layer.name}")
        print(f"  Type: {type(layer)}")
        if hasattr(layer, 'function'):
            print(f"  Function: {layer.function}")

        # Test stop_gradient functionality
        test_input = tf.constant([[1.0, 2.0, 3.0]])
        test_output = layer(test_input)

        # Check if gradients are stopped
        with tf.GradientTape() as tape:
            tape.watch(test_input)
            output = layer(test_input)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, test_input)
        if grads is None:
            print("  ✓ stop_gradient is WORKING (gradients are None)")
        else:
            print(f"  ✗ stop_gradient is NOT working (gradients exist: {grads})")

# Check for Concatenate layer (Task A → Task B connection)
print("\n4. Checking Task A → Task B connection:")
for layer in model.layers:
    if isinstance(layer, keras.layers.Concatenate):
        print(f"Found Concatenate layer: {layer.name}")
        print(f"  Inputs: {layer.input}")

# Load data to test predictions
print("\n5. Loading validation data...")
try:
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)

    X_val = data['X_val']
    y_val_a = data['y_val_A']
    y_val_b = data['y_val_B']
    y_val_c = data['y_val_C']

    # Preprocess
    X_val = X_val.reshape(-1, 32, 32, 1).astype(np.float32)
    train_mean = data['X_train'].mean()
    train_std = data['X_train'].std()
    X_val = (X_val - train_mean) / (train_std + 1e-6)

    print(f"✓ Loaded {len(X_val)} validation samples")
    print(f"  Data shape: {X_val.shape}")
    print(f"  Data range: [{X_val.min():.3f}, {X_val.max():.3f}]")

    # Make predictions
    print("\n6. Testing predictions...")
    preds = model.predict(X_val[:100], verbose=0)
    pred_a, pred_b, pred_c = preds

    print(f"Task A predictions shape: {pred_a.shape}")
    print(f"Task B predictions shape: {pred_b.shape}")
    print(f"Task C predictions shape: {pred_c.shape}")

    # Check if predictions are reasonable
    print("\n7. Prediction Statistics:")
    print(f"Task A max prob: {pred_a.max():.4f}, min prob: {pred_a.min():.4f}")
    print(f"Task B max prob: {pred_b.max():.4f}, min prob: {pred_b.min():.4f}")
    print(f"Task C values: [{pred_c.min():.4f}, {pred_c.max():.4f}]")

    # Calculate accuracy
    acc_a = (pred_a.argmax(axis=1) == y_val_a[:100]).mean()
    acc_b = (pred_b.argmax(axis=1) == y_val_b[:100]).mean()
    mae_c = np.abs(pred_c.flatten() - y_val_c[:100]).mean()

    print(f"\n8. Accuracy on first 100 validation samples:")
    print(f"Task A: {acc_a*100:.2f}%")
    print(f"Task B: {acc_b*100:.2f}%")
    print(f"Task C MAE: {mae_c:.4f}")

    # Check if model is just guessing randomly
    print(f"\n9. Random Baseline Comparison:")
    print(f"Task A random: {100/10:.2f}% (10 classes)")
    print(f"Task B random: {100/32:.2f}% (32 classes)")

    if acc_a < 0.15:
        print("\n⚠️  WARNING: Task A accuracy is near random! Model may not have trained properly.")
    if acc_b < 0.05:
        print("⚠️  WARNING: Task B accuracy is near random! Model may not have trained properly.")

except Exception as e:
    print(f"✗ Failed to load/test data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Debug analysis complete!")
print("="*70)
