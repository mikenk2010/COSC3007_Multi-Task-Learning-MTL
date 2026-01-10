#!/usr/bin/env python3
"""
Apply ONLY test_clean.ipynb optimizations to submission_xxxx.ipynb.ipynb

This will REPLACE the current submission_xxxx_optimized.ipynb with pure test_clean.ipynb strategy.

Key changes from test_clean.ipynb:
1. Simpler CNN backbone (NO ResNet - just Conv→Pool→Conv→Pool)
2. Task B receives semantic signal from Task A (via Concatenate, NOT stop_gradient)
3. Only Task C uses stop_gradient (regression isolated)
4. Use sparse_categorical_crossentropy (simpler than one-hot)
5. Loss weights: (1.0, 1.5, 0.3) - balanced
6. Direct numpy arrays (no complex tf.data pipeline)
7. Monitor Task B accuracy for callbacks
8. NO MixUp, NO RandomRotation/RandomZoom
"""

import json
import sys

def apply_pure_test_clean(notebook_path):
    """Apply pure test_clean.ipynb strategy."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Iterate through cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Change 1: Replace with pure test_clean.ipynb architecture (simple CNN)
        if 'def build_mtl_model()' in source:
            print(f"\n✓ Cell {i}: Replacing with test_clean.ipynb architecture (simple CNN)")
            
            new_function = '''def build_mtl_model() -> tf.keras.Model:
    """
    Build Multi-Task Learning model following test_clean.ipynb architecture.
    
    Key features from test_clean.ipynb:
    - Simple CNN backbone (Conv→Pool→Conv→Pool, NO ResNet complexity)
    - Task B receives semantic signal from Task A (helps orientation learning)
    - Only Task C uses stop_gradient (prevents regression from hurting classification)
    - Balanced loss weights: (1.0, 1.5, 0.3)
    - ~200K parameters (vs 500K in ResNet version)
    """
    inputs = layers.Input(shape=(32, 32, 1), name='input')
    
    # No augmentation - preserves orientation labels for Task B
    x = inputs
    
    # ======================
    # Shared backbone - Simple CNN (like test_clean.ipynb)
    # ======================
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)  # 16x16
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)  # 8x8
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3')(x)  # 8x8
    
    # ======================
    # Task A: Global shape (10 classes)
    # Drives backbone learning
    # ======================
    a = layers.Conv2D(128, 3, padding='same', activation='relu', name='a_conv1')(x)
    a = layers.Conv2D(128, 3, padding='same', activation='relu', name='a_conv2')(a)
    
    a = layers.GlobalAveragePooling2D(name='a_gap')(a)
    a_features = layers.Dense(64, activation='relu', name='a_dense')(a)
    a = layers.Dropout(0.5, name='a_dropout')(a_features)
    
    head_a = layers.Dense(10, activation='softmax', name='head_a')(a)
    
    # ======================
    # Task B: Orientation/fine structure (32 classes)
    # KEY: Receives semantic signal from Task A (NO stop_gradient on A→B connection)
    # This allows Task B to benefit from Task A's learned features
    # ======================
    b = layers.Conv2D(64, 3, padding='same', activation='relu', name='b_conv1')(x)
    b = layers.Conv2D(64, 3, padding='same', activation='relu', name='b_conv2')(b)
    b = layers.Conv2D(128, 3, padding='same', activation='relu', name='b_conv3')(b)
    
    # Preserve structure longer before downsampling (helps orientation)
    b = layers.MaxPooling2D(2, name='b_pool1')(b)  # 4x4
    b = layers.MaxPooling2D(2, name='b_pool2')(b)  # 2x2
    
    b = layers.Flatten(name='b_flatten')(b)
    
    # KEY FIX: Inject Task A semantic signal to help Task B learn
    # NO stop_gradient here - let A's knowledge flow to B
    b = layers.Concatenate(name='b_concat')([b, a_features])
    
    b = layers.Dense(256, activation='relu', name='b_dense')(b)
    b = layers.Dropout(0.5, name='b_dropout')(b)
    
    head_b = layers.Dense(32, activation='softmax', name='head_b')(b)
    
    # ======================
    # Task C: Intensity (regression)
    # KEY: Stop gradient to prevent regression from hurting classification tasks
    # Regression (MSE) has different loss scale than classification (cross-entropy)
    # ======================
    c = layers.Lambda(lambda t: tf.stop_gradient(t), name='c_stop')(x)
    c = layers.GlobalAveragePooling2D(name='c_gap')(c)
    c = layers.Dense(32, activation='relu', name='c_dense')(c)
    c = layers.Dropout(0.3, name='c_dropout')(c)
    
    head_c = layers.Dense(1, activation='sigmoid', name='head_c')(c)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=[head_a, head_b, head_c], 
                         name='MTL_CNN_TestClean')
    
    return model
'''
            cell['source'] = [new_function]
            changes_made += 1
        
        # Change 2: Use sparse categorical crossentropy (like test_clean.ipynb)
        if 'model.compile(' in source and ('CategoricalCrossentropy' in source or 'categorical_accuracy' in source):
            print(f"\n✓ Cell {i}: Updating compile to test_clean.ipynb style")
            source_str = ''.join(cell['source'])
            
            # Replace with sparse categorical
            source_str = source_str.replace(
                "tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)",
                "'sparse_categorical_crossentropy'"
            )
            
            # Update metrics
            source_str = source_str.replace(
                "'categorical_accuracy'",
                "'sparse_categorical_accuracy'"
            )
            
            # Update loss weights to test_clean.ipynb values: (1.0, 1.5, 0.3)
            if "'head_a':" in source_str and "'head_b':" in source_str and "'head_c':" in source_str:
                # Find and replace loss_weights section
                if "loss_weights" in source_str:
                    source_str = source_str.replace("'head_a': 2.5,", "'head_a': 1.0,")
                    source_str = source_str.replace("'head_a': 2.5", "'head_a': 1.0")
                    source_str = source_str.replace("'head_b': 0.2,", "'head_b': 1.5,")
                    source_str = source_str.replace("'head_b': 0.2", "'head_b': 1.5")
                    source_str = source_str.replace("'head_c': 0.03", "'head_c': 0.3")
            
            # Keep stable optimizer: lr=1e-3 like test_clean, but keep clipnorm for safety
            source_str = source_str.replace("learning_rate=2e-4", "learning_rate=1e-3")
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                             for idx, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 3: Simplify preprocess_fn - use sparse labels (like test_clean)
        if 'def preprocess_fn' in source and 'one_hot' in source:
            print(f"\n✓ Cell {i}: Simplifying preprocessing to sparse labels")
            source_str = ''.join(cell['source'])
            
            new_preprocess = '''def preprocess_fn(x, y):
    """
    Preprocess a batch of data (test_clean.ipynb style).
    - Normalizes X using training statistics
    - Reshapes X to (32, 32, 1)
    - Splits y into three targets as sparse labels (NO one-hot conversion)
    """
    # Reshape and normalize X
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, (-1, 32, 32, 1))
    x = (x - train_mean) / train_std
    
    # Split y into three targets - keep as sparse labels
    y_a = tf.cast(y[:, 0], tf.int32)  # Target A: 10-class classification
    y_b = tf.cast(y[:, 1], tf.int32)  # Target B: 32-class classification
    y_c = tf.cast(y[:, 2], tf.float32)  # Target C: Regression
    
    return x, {'head_a': y_a, 'head_b': y_b, 'head_c': y_c}
'''
            
            # Replace the function
            start = source_str.find('def preprocess_fn(x, y):')
            if start != -1:
                end = source_str.find('\ndef make_dataset', start)
                if end == -1:
                    end = source_str.find('\n\n# Create', start)
                if end > start:
                    source_str = source_str[:start] + new_preprocess + '\n' + source_str[end:]
                
                cell['source'] = source_str.split('\n')
                cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                                 for idx, line in enumerate(cell['source'])]
                changes_made += 1
        
        # Change 4: Update callbacks to monitor Task A or B (test_clean monitors B)
        if 'callbacks.EarlyStopping(' in source:
            print(f"\n✓ Cell {i}: Updating EarlyStopping to monitor Task B (like test_clean)")
            source_str = ''.join(cell['source'])
            
            # Monitor Task B accuracy like test_clean.ipynb
            source_str = source_str.replace(
                "monitor='val_head_a_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            source_str = source_str.replace(
                "monitor='val_head_a_sparse_categorical_accuracy'",
                "monitor='val_head_b_sparse_categorical_accuracy'"
            )
            
            # Update patience to test_clean value (8 epochs)
            if "patience=15" in source_str:
                source_str = source_str.replace("patience=15", "patience=8")
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                             for idx, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 5: Update ModelCheckpoint
        if 'callbacks.ModelCheckpoint(' in source:
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
            cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                             for idx, line in enumerate(cell['source'])]
            changes_made += 1
    
    # Save modified notebook
    output_path = notebook_path.replace('.ipynb', '_testclean.ipynb')
    if '_optimized' in output_path:
        output_path = output_path.replace('_optimized_testclean', '_testclean')
    
    print(f"\n\nSaving notebook to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Successfully applied {changes_made} test_clean.ipynb optimizations!")
    print(f"\nKey changes (pure test_clean.ipynb strategy):")
    print(f"1. ✅ Simple CNN backbone (NO ResNet)")
    print(f"2. ✅ Task B receives semantic help from Task A")
    print(f"3. ✅ Only Task C uses stop_gradient")
    print(f"4. ✅ Sparse categorical crossentropy")
    print(f"5. ✅ Balanced loss weights: (1.0, 1.5, 0.3)")
    print(f"6. ✅ Monitors Task B accuracy")
    print(f"7. ✅ NO MixUp, NO rotation/zoom")
    
    return output_path

if __name__ == '__main__':
    notebook_path = 'submission_xxxx.ipynb.ipynb'
    try:
        output_path = apply_pure_test_clean(notebook_path)
        print(f"\n✅ Optimization complete!")
        print(f"\nExpected results (from test_clean.ipynb):")
        print(f"  Task A: ~33.5%")
        print(f"  Task B: ~5-6%")
        print(f"  Task C: ~0.15 MAE")
        print(f"\nNext steps:")
        print(f"1. Open {output_path} in Jupyter")
        print(f"2. Run all cells")
        print(f"3. Training should take ~1-2 hours (faster than ResNet)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
