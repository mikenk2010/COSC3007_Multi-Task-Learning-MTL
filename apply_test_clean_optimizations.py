#!/usr/bin/env python3
"""
Apply test_clean.ipynb optimizations to submission_xxxx_optimized.ipynb

Key changes from test_clean.ipynb:
1. Simpler CNN backbone (faster convergence, fewer parameters)
2. Task B receives semantic signal from Task A (via Concatenate)
3. Only Task C uses stop_gradient
4. Use sparse_categorical_crossentropy (simpler than one-hot)
5. Adjusted loss weights: (1.0, 1.5, 0.3)
6. Simplified data pipeline (direct numpy, no tf.data overhead)
7. Monitor Task B accuracy for callbacks
"""

import json
import sys

def apply_test_clean_optimizations(notebook_path):
    """Apply test_clean.ipynb strategies to the notebook."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Iterate through cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Change 1: Replace build_mtl_model with test_clean.ipynb architecture
        if 'def build_mtl_model()' in source:
            print(f"\n✓ Cell {i}: Replacing architecture with test_clean.ipynb style")
            
            new_function = '''def build_mtl_model() -> tf.keras.Model:
    """
    Build Multi-Task Learning model following test_clean.ipynb architecture.
    
    Key features:
    - Simpler CNN backbone (faster convergence)
    - Task B receives semantic signal from Task A
    - Only Task C uses stop_gradient (prevents regression interference)
    - Balanced loss weights for all tasks
    """
    inputs = layers.Input(shape=(32, 32, 1), name='input')
    
    # No augmentation (preserves orientation labels)
    x = inputs
    
    # ======================
    # Shared backbone - Simple CNN
    # ======================
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)  # 16x16
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)  # 8x8
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3')(x)  # 8x8
    
    # ======================
    # Task A: Global shape (10 classes)
    # ======================
    a = layers.Conv2D(128, 3, padding='same', activation='relu', name='a_conv1')(x)
    a = layers.Conv2D(128, 3, padding='same', activation='relu', name='a_conv2')(a)
    
    a = layers.GlobalAveragePooling2D(name='a_gap')(a)
    a_features = layers.Dense(64, activation='relu', name='a_dense')(a)
    a = layers.Dropout(0.5, name='a_dropout')(a_features)
    
    head_a = layers.Dense(10, activation='softmax', name='head_a')(a)
    
    # ======================
    # Task B: Orientation/fine structure (32 classes)
    # Key: Receives semantic signal from Task A
    # ======================
    b = layers.Conv2D(64, 3, padding='same', activation='relu', name='b_conv1')(x)
    b = layers.Conv2D(64, 3, padding='same', activation='relu', name='b_conv2')(b)
    b = layers.Conv2D(128, 3, padding='same', activation='relu', name='b_conv3')(b)
    
    # Preserve structure longer before downsampling
    b = layers.MaxPooling2D(2, name='b_pool1')(b)  # 4x4
    b = layers.MaxPooling2D(2, name='b_pool2')(b)  # 2x2
    
    b = layers.Flatten(name='b_flatten')(b)
    
    # KEY: Inject Task A semantic signal to help Task B
    b = layers.Concatenate(name='b_concat')([b, a_features])
    
    b = layers.Dense(256, activation='relu', name='b_dense')(b)
    b = layers.Dropout(0.5, name='b_dropout')(b)
    
    head_b = layers.Dense(32, activation='softmax', name='head_b')(b)
    
    # ======================
    # Task C: Intensity (regression)
    # KEY: Stop gradient to prevent regression from hurting classification
    # ======================
    c = layers.Lambda(lambda t: tf.stop_gradient(t), name='c_stop')(x)
    c = layers.GlobalAveragePooling2D(name='c_gap')(c)
    c = layers.Dense(32, activation='relu', name='c_dense')(c)
    c = layers.Dropout(0.3, name='c_dropout')(c)
    
    head_c = layers.Dense(1, activation='sigmoid', name='head_c')(c)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=[head_a, head_b, head_c], 
                         name='MTL_CNN')
    
    return model
'''
            cell['source'] = [new_function]
            changes_made += 1
        
        # Change 2: Simplify compile - use sparse categorical crossentropy
        if 'model.compile(' in source and 'CategoricalCrossentropy' in source and 'label_smoothing' in source:
            print(f"\n✓ Cell {i}: Simplifying compile with sparse_categorical_crossentropy")
            source_str = ''.join(cell['source'])
            
            # Replace CategoricalCrossentropy with sparse version
            source_str = source_str.replace(
                "tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)",
                "'sparse_categorical_crossentropy'"
            )
            
            # Update metrics
            source_str = source_str.replace(
                "'categorical_accuracy'",
                "'sparse_categorical_accuracy'"
            )
            
            # Update loss weights to test_clean.ipynb values
            if "'head_a': 2.5" in source_str:
                source_str = source_str.replace("'head_a': 2.5,", "'head_a': 1.0,")
                source_str = source_str.replace("'head_b': 0.2,", "'head_b': 1.5,")
                source_str = source_str.replace("'head_c': 0.03", "'head_c': 0.3")
            
            # Keep the lower LR and clipnorm
            # source_str already has learning_rate=2e-4, clipnorm=1.0
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 3: Update data pipeline to use direct numpy (simpler, like test_clean)
        if 'def preprocess_fn' in source and 'tf.one_hot' in source:
            print(f"\n✓ Cell {i}: Simplifying data preprocessing (remove one-hot conversion)")
            source_str = ''.join(cell['source'])
            
            # Simplify preprocess_fn to NOT convert to one-hot
            new_preprocess = '''def preprocess_fn(x, y):
    """
    Preprocess a batch of data.
    - Normalizes X using training statistics
    - Reshapes X to (32, 32, 1)
    - Splits y into three targets (keeps as sparse labels)
    """
    # Reshape and normalize X
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, (-1, 32, 32, 1))
    x = (x - train_mean) / train_std
    
    # Split y into three targets - keep as sparse labels for sparse_categorical_crossentropy
    y_a = tf.cast(y[:, 0], tf.int32)  # Target A: 10-class classification
    y_b = tf.cast(y[:, 1], tf.int32)  # Target B: 32-class classification
    y_c = tf.cast(y[:, 2], tf.float32)  # Target C: Regression
    
    return x, {'head_a': y_a, 'head_b': y_b, 'head_c': y_c}
'''
            if 'def preprocess_fn(x, y):' in source_str:
                # Replace the function
                start = source_str.find('def preprocess_fn(x, y):')
                end = source_str.find('\ndef make_dataset', start)
                if end == -1:
                    end = source_str.find('\n\n# ', start)
                if end > start:
                    source_str = source_str[:start] + new_preprocess + '\n' + source_str[end:]
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 4: Update callback monitors to Task B (like test_clean)
        if 'callbacks.EarlyStopping(' in source and 'head_a_categorical_accuracy' in source:
            print(f"\n✓ Cell {i}: Updating callbacks to monitor Task B (like test_clean)")
            source_str = ''.join(cell['source'])
            source_str = source_str.replace(
                "monitor='val_head_a_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            source_str = source_str.replace(
                "monitor='val_head_a_sparse_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 5: Update ModelCheckpoint monitor
        if 'callbacks.ModelCheckpoint(' in source and 'head_a_categorical_accuracy' in source:
            print(f"\n✓ Cell {i}: Updating ModelCheckpoint to monitor Task B")
            source_str = ''.join(cell['source'])
            source_str = source_str.replace(
                "monitor='val_head_a_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            source_str = source_str.replace(
                "monitor='val_head_a_sparse_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
    
    # Save modified notebook
    output_path = notebook_path.replace('_optimized.ipynb', '_test_clean_optimized.ipynb')
    print(f"\n\nSaving notebook to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Successfully applied {changes_made} test_clean.ipynb optimizations!")
    print(f"\nKey changes from test_clean.ipynb:")
    print(f"1. Simpler CNN backbone (no ResNet complexity)")
    print(f"2. Task B receives semantic features from Task A")
    print(f"3. Only Task C uses stop_gradient")
    print(f"4. Sparse categorical crossentropy (no one-hot)")
    print(f"5. Balanced loss weights: (1.0, 1.5, 0.3)")
    print(f"6. Monitors Task B accuracy for callbacks")
    
    return output_path

if __name__ == '__main__':
    notebook_path = 'submission_xxxx_optimized.ipynb'
    try:
        output_path = apply_test_clean_optimizations(notebook_path)
        print(f"\n✅ Optimization complete!")
        print(f"\nNext steps:")
        print(f"1. Open {output_path} in Jupyter")
        print(f"2. Run all cells")
        print(f"3. This combines best of both test_clean.ipynb and test_clean2.ipynb strategies")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
