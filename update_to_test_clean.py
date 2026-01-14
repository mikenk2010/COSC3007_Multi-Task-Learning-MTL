#!/usr/bin/env python3
"""
Update submission_s3715228_s3343711_s4139514.ipynb to match test_clean.ipynb exactly.

Key changes:
1. Replace tf.data.Dataset with direct numpy arrays (test_clean.ipynb approach)
2. Remove clipnorm from optimizer
3. Update epochs to 80 (from 50)
4. Update training code to use direct numpy arrays
5. Match test_clean.ipynb training approach exactly
"""

import json
import sys

def update_notebook(notebook_path):
    """Update notebook to match test_clean.ipynb."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Iterate through cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Change 1: Replace data preprocessing to use direct numpy arrays (test_clean.ipynb)
        if 'train_ds = make_dataset' in source and 'X_train_mtl' not in source:
            print(f"\n✓ Cell {i}: Replacing tf.data.Dataset with direct numpy arrays (test_clean.ipynb)")
            
            new_code = '''# Prepare data for multi-task training (test_clean.ipynb approach)
# Use direct numpy arrays instead of tf.data.Dataset for simplicity
X_train_mtl = X_train[..., None].astype('float32')
X_val_mtl = X_val[..., None].astype('float32')

# Normalize (test_clean.ipynb approach)
mean = X_train_mtl.mean()
std = X_train_mtl.std() + 1e-6  # Add epsilon for numerical stability
X_train_mtl = (X_train_mtl - mean) / std
X_val_mtl = (X_val_mtl - mean) / std

# Extract targets (test_clean.ipynb approach)
y_A_train, y_B_train, y_C_train = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_A_val, y_B_val, y_C_val = y_val[:, 0], y_val[:, 1], y_val[:, 2]

print("Data prepared for multi-task learning (test_clean.ipynb style):")
print(f"  X_train: {X_train_mtl.shape}")
print(f"  X_val: {X_val_mtl.shape}")
print(f"  Target A: {y_A_train.shape} (10 classes)")
print(f"  Target B: {y_B_train.shape} (32 classes)")
print(f"  Target C: {y_C_train.shape} (regression [{y_C_train.min():.4f}, {y_C_train.max():.4f}])")
'''
            cell['source'] = [new_code]
            changes_made += 1
        
        # Change 2: Remove clipnorm from optimizer
        if 'clipnorm=1.0' in source:
            print(f"\n✓ Cell {i}: Removing clipnorm from optimizer (test_clean.ipynb)")
            cell['source'] = [line.replace('clipnorm=1.0,', '').replace('clipnorm=1.0', '') 
                             for line in cell['source']]
            changes_made += 1
        
        # Change 3: Update epochs to 80
        if 'EPOCHS = 50' in source or 'epochs=50' in source:
            print(f"\n✓ Cell {i}: Updating epochs to 80 (test_clean.ipynb)")
            cell['source'] = [line.replace('EPOCHS = 50', 'EPOCHS = 80')
                             .replace('epochs=50', 'epochs=80')
                             for line in cell['source']]
            changes_made += 1
        
        # Change 4: Update training code to use direct numpy arrays
        if 'model.fit(' in source and 'train_ds' in source:
            print(f"\n✓ Cell {i}: Updating training code to use direct numpy arrays (test_clean.ipynb)")
            # Replace the fit call
            new_fit = '''    history = model.fit(
        X_train_mtl,
        {'head_a': y_A_train, 'head_b': y_B_train, 'head_c': y_C_train},
        validation_data=(
            X_val_mtl,
            {'head_a': y_A_val, 'head_b': y_B_val, 'head_c': y_C_val}
        ),
        epochs=80,
        batch_size=64,
        callbacks=callbacks_list,
        verbose=2
    )'''
            
            # Find and replace the fit call
            source_str = ''.join(cell['source'])
            start = source_str.find('history = model.fit(')
            if start == -1:
                start = source_str.find('model.fit(')
            if start != -1:
                # Find the end of the fit call
                end = source_str.find('\n    )', start)
                if end == -1:
                    end = source_str.find('\n)', start)
                if end != -1:
                    end += len('\n    )')
                    source_str = source_str[:start] + new_fit + source_str[end:]
                    cell['source'] = source_str.split('\n')
                    cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                                     for idx, line in enumerate(cell['source'])]
                    changes_made += 1
    
    # Save updated notebook
    if changes_made > 0:
        print(f"\n✓ Made {changes_made} changes. Saving notebook...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Notebook updated successfully!")
    else:
        print("\n⚠ No changes made. Notebook may already be updated.")
    
    return changes_made

if __name__ == '__main__':
    notebook_path = 'submission_s3715228_s3343711_s4139514.ipynb'
    update_notebook(notebook_path)
