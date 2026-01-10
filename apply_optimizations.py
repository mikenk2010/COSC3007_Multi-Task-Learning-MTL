#!/usr/bin/env python3
"""
Script to apply optimizations from test_clean.ipynb and test_clean2.ipynb
to submission_xxxx.ipynb.ipynb

This script will:
1. Disable MixUp
2. Update build_mtl_model() to use stop_gradient branches
3. Fix loss weights (2.5, 0.2, 0.03 instead of 1.0, 2.5, 10.0)
4. Update optimizer (lr=2e-4, clipnorm=1.0)
5. Fix callbacks to monitor Task A accuracy
"""

import json
import sys

def apply_optimizations(notebook_path):
    """Apply all optimizations to the notebook."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Iterate through cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Change 1: Disable MixUp
        if 'use_mixup=True' in source:
            print(f"\n✓ Cell {i}: Disabling MixUp")
            cell['source'] = [line.replace('use_mixup=True', 'use_mixup=False') 
                             for line in cell['source']]
            changes_made += 1
        
        # Change 2: Replace build_mtl_model function
        if 'def build_mtl_model()' in source and 'RandomRotation' in source:
            print(f"\n✓ Cell {i}: Replacing build_mtl_model() with stop-gradient version")
            
            new_function = '''def build_mtl_model() -> tf.keras.Model:
    """
    Build Multi-Task Learning model with stop-gradient branches.
    Prevents negative transfer between tasks.
    """
    inputs = layers.Input(shape=(32, 32, 1), name='input')
    
    # Remove harmful RandomRotation/RandomZoom
    x = inputs
    
    # Initial convolution
    x = layers.SeparableConv2D(64, 3, padding='same', name='initial_conv')(x)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(name='initial_relu')(x)
    x = layers.MaxPooling2D(2, name='initial_pool')(x)
    
    # Residual blocks
    x = residual_block(x, 64, stride=1, name_prefix='res_block1')
    x = residual_block(x, 64, stride=1, name_prefix='res_block2')
    x = residual_block(x, 128, stride=2, name_prefix='res_block3')
    x = residual_block(x, 128, stride=1, name_prefix='res_block4')
    
    # Head A: Primary task (drives backbone)
    a = layers.SeparableConv2D(128, 3, padding='same', name='a_conv')(x)
    a = layers.BatchNormalization(name='a_bn')(a)
    a = layers.ReLU(name='a_relu')(a)
    a_semantic = layers.Conv2D(64, 1, activation='relu', name='a_semantic')(a)
    
    a = layers.GlobalAveragePooling2D(name='global_avg_pool')(a)
    shared_features = layers.Dense(256, activation='relu', name='shared_dense')(a)
    shared_features = layers.BatchNormalization(name='shared_bn')(shared_features)
    
    head_a = layers.Dense(128, activation='relu', name='head_a_dense1')(shared_features)
    head_a = layers.Dense(10, activation='softmax', name='head_a')(head_a)
    
    # Head B: Stop gradient to prevent negative transfer
    b = layers.Lambda(lambda t: tf.stop_gradient(t), name='b_stop')(x)
    b = layers.SeparableConv2D(128, 3, padding='same', name='b_conv1')(b)
    b = layers.BatchNormalization(name='b_bn1')(b)
    b = layers.ReLU(name='b_relu1')(b)
    b = layers.SeparableConv2D(192, 3, padding='same', name='b_conv2')(b)
    b = layers.BatchNormalization(name='b_bn2')(b)
    b = layers.ReLU(name='b_relu2')(b)
    
    a_sem_stopped = layers.Lambda(lambda t: tf.stop_gradient(t), name='a_sem_stop')(a_semantic)
    b = layers.Concatenate(name='b_concat')([b, a_sem_stopped])
    
    b = layers.MaxPooling2D(2, name='b_pool')(b)
    b = layers.Flatten(name='b_flatten')(b)
    b = layers.Dense(256, activation='relu', name='head_b_dense1')(b)
    b = layers.Dropout(0.5, name='head_b_dropout')(b)
    head_b = layers.Dense(32, activation='softmax', name='head_b')(b)
    
    # Head C: Stop gradient
    c = layers.Lambda(lambda t: tf.stop_gradient(t), name='c_stop')(x)
    c = layers.GlobalAveragePooling2D(name='c_gap')(c)
    c = layers.Dense(64, activation='relu', name='head_c_dense1')(c)
    head_c = layers.Dense(1, activation='sigmoid', name='head_c')(head_c)
    
    model = models.Model(inputs=inputs, outputs=[head_a, head_b, head_c], name='MTL_ResNet')
    return model
'''
            cell['source'] = [new_function]
            changes_made += 1
        
        # Change 3: Fix loss weights in model.compile()
        if 'model.compile(' in source and "'head_c': 10.0" in source:
            print(f"\n✓ Cell {i}: Fixing loss weights and optimizer")
            source_str = ''.join(cell['source'])
            
            # Update learning rate
            source_str = source_str.replace('learning_rate=1e-3', 'learning_rate=2e-4, clipnorm=1.0')
            
            # Update loss weights
            source_str = source_str.replace("'head_a': 1.0,", "'head_a': 2.5,")
            source_str = source_str.replace("'head_b': 2.5,", "'head_b': 0.2,")
            source_str = source_str.replace("'head_c': 10.0", "'head_c': 0.03")
            
            # Update print statements
            source_str = source_str.replace(
                'print("  head_a: 1.0")',
                'print("  head_a: 2.5 (primary task)")'
            )
            source_str = source_str.replace(
                'print("  head_b: 2.5 (difficult 32-class task)")',
                'print("  head_b: 0.2 (low weight to prevent negative transfer)")'
            )
            source_str = source_str.replace(
                'print("  head_c: 10.0 (MSE scale compensation)")',
                'print("  head_c: 0.03 (minimal weight)")'
            )
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 4: Fix EarlyStopping monitor
        if "callbacks.EarlyStopping(" in source and "monitor='val_loss'" in source:
            print(f"\n✓ Cell {i}: Updating EarlyStopping to monitor Task A accuracy")
            source_str = ''.join(cell['source'])
            source_str = source_str.replace(
                "monitor='val_loss',",
                "monitor='val_head_a_categorical_accuracy',\n        mode='max',"
            )
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 5: Fix ModelCheckpoint monitor
        if "callbacks.ModelCheckpoint(" in source and "monitor='val_loss'" in source:
            print(f"\n✓ Cell {i}: Updating ModelCheckpoint to monitor Task A accuracy")
            source_str = ''.join(cell['source'])
            source_str = source_str.replace(
                "monitor='val_loss',",
                "monitor='val_head_a_categorical_accuracy',\n        mode='max',"
            )
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
        
        # Change 6: Fix ensemble training compile
        if 'initial_lr = 1e-3' in source and 'CosineDecay' in source:
            print(f"\n✓ Cell {i}: Updating ensemble training optimizer")
            source_str = ''.join(cell['source'])
            source_str = source_str.replace('initial_lr = 1e-3', 'initial_lr = 2e-4')
            source_str = source_str.replace(
                'optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay)',
                'optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay, clipnorm=1.0)'
            )
            # Update loss weights
            source_str = source_str.replace("'head_a': 1.0,", "'head_a': 2.5,")
            source_str = source_str.replace("'head_b': 2.5,", "'head_b': 0.2,")
            source_str = source_str.replace("'head_c': 10.0", "'head_c': 0.03")
            
            cell['source'] = source_str.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                             for i, line in enumerate(cell['source'])]
            changes_made += 1
    
    # Save modified notebook
    output_path = notebook_path.replace('.ipynb', '_optimized.ipynb')
    print(f"\n\nSaving optimized notebook to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Successfully applied {changes_made} optimizations!")
    print(f"\nNext steps:")
    print(f"1. Open {output_path} in Jupyter")
    print(f"2. Run cells from the data pipeline onwards")
    print(f"3. Expected improvements:")
    print(f"   - Task A (val): ~30-36% (up from ~12-15%)")
    print(f"   - Task B (val): ~5-7% (up from ~3-5%)")
    print(f"   - Task C (val): ~0.16-0.19 MAE (improved from ~0.23-0.25)")
    
    return output_path

if __name__ == '__main__':
    notebook_path = 'submission_xxxx.ipynb.ipynb'
    try:
        output_path = apply_optimizations(notebook_path)
        print(f"\n✅ Optimization complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
