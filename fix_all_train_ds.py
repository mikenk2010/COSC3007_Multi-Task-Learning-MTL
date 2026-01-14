#!/usr/bin/env python3
"""
Fix ALL train_ds and val_ds references in the notebook.
Replace with numpy arrays (X_train_mtl, X_val_mtl, etc.)
"""

import json
import re

def fix_notebook(notebook_path):
    """Fix all train_ds/val_ds references."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        original_source = source
        
        # Fix 1: Replace train_ds in total_steps calculation
        if 'total_steps = len(list(train_ds))' in source:
            print(f"\n✓ Cell {i}: Fixing total_steps calculation")
            source = re.sub(
                r'total_steps = len\(list\(train_ds\)\) \* \d+.*',
                'batch_size = 64\n        epochs = 80\n        steps_per_epoch = len(X_train_mtl) // batch_size\n        total_steps = steps_per_epoch * epochs',
                source
            )
        
        # Fix 2: Replace train_ds in model.fit()
        if 'model.fit(' in source and 'train_ds' in source:
            print(f"✓ Cell {i}: Fixing model.fit() to use numpy arrays")
            # Replace the fit call
            source = re.sub(
                r'history = model\.fit\(\s*train_ds,\s*validation_data=val_ds,',
                '''history = model.fit(
            X_train_mtl,
            {'head_a': y_A_train, 'head_b': y_B_train, 'head_c': y_C_train},
            validation_data=(
                X_val_mtl,
                {'head_a': y_A_val, 'head_b': y_B_val, 'head_c': y_C_val}
            ),''',
                source,
                flags=re.MULTILINE
            )
        
        # Fix 3: Replace val_ds in model.evaluate()
        if 'model.evaluate(val_ds' in source:
            print(f"✓ Cell {i}: Fixing model.evaluate() to use numpy arrays")
            source = source.replace(
                'model.evaluate(val_ds, verbose=1)',
                "model.evaluate(X_val_mtl, {'head_a': y_A_val, 'head_b': y_B_val, 'head_c': y_C_val}, verbose=1)"
            )
            source = source.replace(
                'model.evaluate(val_ds, verbose=0)',
                "model.evaluate(X_val_mtl, {'head_a': y_A_val, 'head_b': y_B_val, 'head_c': y_C_val}, verbose=0)"
            )
        
        # Fix 4: Replace val_ds in model.predict()
        if 'model.predict(val_ds' in source and 'ensemble' in source.lower():
            print(f"✓ Cell {i}: Fixing model.predict() to use numpy arrays")
            source = source.replace(
                'model.predict(val_ds, verbose=0)',
                'model.predict(X_val_mtl, verbose=0)'
            )
            source = source.replace(
                'model.predict(val_ds, verbose=1)',
                'model.predict(X_val_mtl, verbose=1)'
            )
        
        # Fix 5: Remove CosineDecay and use simple LR (matches test_clean.ipynb)
        if 'cosine_decay = tf.keras.optimizers.schedules.CosineDecay' in source:
            print(f"✓ Cell {i}: Simplifying to use simple LR (test_clean.ipynb approach)")
            # Remove cosine_decay setup
            source = re.sub(
                r'# UPGRADE: Cosine Decay Learning Rate Schedule.*?alpha=0\.1.*?\n',
                '',
                source,
                flags=re.DOTALL
            )
            # Replace optimizer with simple LR
            source = source.replace(
                'optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),',
                'optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),'
            )
            # Add initial_lr if not present
            if 'initial_lr = 1e-3' not in source:
                source = source.replace(
                    'model.compile(',
                    '        initial_lr = 1e-3\n        model.compile(',
                    1
                )
        
        # Fix 6: Update print statements
        if 'Using CosineDecay LR schedule' in source:
            print(f"✓ Cell {i}: Updating print statements")
            source = source.replace(
                'print(f"  Using CosineDecay LR schedule: {initial_lr} → {initial_lr * 0.1} over {total_steps} steps")',
                'print(f"  Using simple LR: {initial_lr} (test_clean.ipynb approach)")'
            )
            source = source.replace(
                'print(f"  Label Smoothing: 0.1 (prevents overconfidence)")',
                ''
            )
        
        # Fix 7: Add batch_size to fit call if missing
        if 'model.fit(' in source and 'batch_size' not in source and 'X_train_mtl' in source:
            print(f"✓ Cell {i}: Adding batch_size to fit call")
            source = re.sub(
                r'(epochs=\d+,)\s*(callbacks=)',
                r'\1\n            batch_size=64,\n            \2',
                source
            )
        
        # Fix 8: Change verbose=1 to verbose=2 for consistency
        if 'model.fit(' in source and 'X_train_mtl' in source and 'verbose=1' in source:
            print(f"✓ Cell {i}: Changing verbose to 2 (test_clean.ipynb)")
            source = source.replace('verbose=1', 'verbose=2', 1)
        
        if source != original_source:
            cell['source'] = source.split('\n')
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
        print("\n⚠ No changes made.")
    
    return changes_made

if __name__ == '__main__':
    notebook_path = 'submission_s3715228_s3343711_s4139514.ipynb'
    fix_notebook(notebook_path)
