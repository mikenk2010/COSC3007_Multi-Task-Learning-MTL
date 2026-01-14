#!/usr/bin/env python3
"""
Fix the NameError: train_ds is not defined in the ensemble training cell.
Replace train_ds references with numpy arrays (X_train_mtl, etc.)
"""

import json

def fix_notebook(notebook_path):
    """Fix train_ds references in ensemble training cell."""
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Find and fix the ensemble training cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Check if this is the ensemble training cell with train_ds error
        if 'total_steps = len(list(train_ds))' in source:
            print(f"\n✓ Cell {i}: Fixing train_ds reference in ensemble training")
            
            # Replace the problematic section
            old_code = """        # UPGRADE: Cosine Decay Learning Rate Schedule (Chapter 13 Optimization)
        # Calculate total training steps for CosineDecay
        total_steps = len(list(train_ds)) * 100  # epochs * steps per epoch
        initial_lr = 1e-3
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=0.1  # Decay to 10% of initial LR
        )
        
        # UPGRADE: Label Smoothing (Chapter 5/13 Regularization)
        # Prevents overconfidence on noisy, limited data
        # Note: With MixUp, labels are one-hot, so we use CategoricalCrossentropy
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
            loss={
                'head_a': 'sparse_categorical_crossentropy',
                'head_b': 'sparse_categorical_crossentropy',
                'head_c': 'mse'
            },
            loss_weights={
                'head_a': 1.0,
                'head_b': 1.5,  # Hardest task (test_clean.ipynb)
                'head_c': 0.3  # Prevent dominance (test_clean.ipynb)
            },
            metrics={
                'head_a': 'sparse_categorical_accuracy',  # Changed from sparse_categorical_accuracy
                'head_b': 'sparse_categorical_accuracy',  # Changed from sparse_categorical_accuracy
                'head_c': ['mse', 'mae']
            }
        )
        
        # Train model
        print(f\"\\nStarting training for Model {i}...\")
        print(f\"  Using CosineDecay LR schedule: {initial_lr} → {initial_lr * 0.1} over {total_steps} steps\")
        print(f\"  Label Smoothing: 0.1 (prevents overconfidence)\")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=80,
            callbacks=callbacks_list,
            verbose=1
        )"""
            
            new_code = """        # Compile model (test_clean.ipynb approach - simple optimizer)
        initial_lr = 1e-3
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),  # Simple LR (matches test_clean)
            loss={
                'head_a': 'sparse_categorical_crossentropy',
                'head_b': 'sparse_categorical_crossentropy',
                'head_c': 'mse'
            },
            loss_weights={
                'head_a': 1.0,
                'head_b': 1.5,  # Hardest task (test_clean.ipynb)
                'head_c': 0.3  # Prevent dominance (test_clean.ipynb)
            },
            metrics={
                'head_a': 'sparse_categorical_accuracy',
                'head_b': 'sparse_categorical_accuracy',
                'head_c': ['mse', 'mae']
            }
        )
        
        # Train model (using direct numpy arrays like test_clean.ipynb)
        print(f\"\\nStarting training for Model {i}...\")
        print(f\"  Using simple LR: {initial_lr} (test_clean.ipynb approach)\")
        history = model.fit(
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
        )"""
            
            # Replace in source
            if old_code in source:
                cell['source'] = source.replace(old_code, new_code).split('\n')
                cell['source'] = [line + '\n' if idx < len(cell['source'])-1 else line 
                                 for idx, line in enumerate(cell['source'])]
                changes_made += 1
            elif 'total_steps = len(list(train_ds))' in source:
                # More flexible replacement if exact match fails
                source_lines = source.split('\n')
                new_lines = []
                skip_until_fit = False
                for line in source_lines:
                    if 'total_steps = len(list(train_ds))' in line:
                        # Skip cosine decay setup, keep simple
                        skip_until_fit = True
                        continue
                    elif skip_until_fit and 'model.compile(' in line:
                        skip_until_fit = False
                        # Insert simple compile
                        new_lines.append("        # Compile model (test_clean.ipynb approach)")
                        new_lines.append("        initial_lr = 1e-3")
                        new_lines.append("        model.compile(")
                        new_lines.append("            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),")
                        new_lines.append("            loss={")
                        new_lines.append("                'head_a': 'sparse_categorical_crossentropy',")
                        new_lines.append("                'head_b': 'sparse_categorical_crossentropy',")
                        new_lines.append("                'head_c': 'mse'")
                        new_lines.append("            },")
                        new_lines.append("            loss_weights={")
                        new_lines.append("                'head_a': 1.0,")
                        new_lines.append("                'head_b': 1.5,")
                        new_lines.append("                'head_c': 0.3")
                        new_lines.append("            },")
                        new_lines.append("            metrics={")
                        new_lines.append("                'head_a': 'sparse_categorical_accuracy',")
                        new_lines.append("                'head_b': 'sparse_categorical_accuracy',")
                        new_lines.append("                'head_c': ['mse', 'mae']")
                        new_lines.append("            }")
                        new_lines.append("        )")
                        new_lines.append("")
                        new_lines.append("        # Train model (using numpy arrays)")
                        new_lines.append("        print(f\"\\nStarting training for Model {i}...\")")
                        new_lines.append("        history = model.fit(")
                        new_lines.append("            X_train_mtl,")
                        new_lines.append("            {'head_a': y_A_train, 'head_b': y_B_train, 'head_c': y_C_train},")
                        new_lines.append("            validation_data=(")
                        new_lines.append("                X_val_mtl,")
                        new_lines.append("                {'head_a': y_A_val, 'head_b': y_B_val, 'head_c': y_C_val}")
                        new_lines.append("            ),")
                        new_lines.append("            epochs=80,")
                        new_lines.append("            batch_size=64,")
                        new_lines.append("            callbacks=callbacks_list,")
                        new_lines.append("            verbose=2")
                        new_lines.append("        )")
                        continue
                    elif skip_until_fit and ('cosine_decay' in line or 'CosineDecay' in line or 'Label Smoothing' in line):
                        continue
                    elif skip_until_fit and 'train_ds' in line:
                        continue
                    elif skip_until_fit and 'val_ds' in line:
                        continue
                    else:
                        new_lines.append(line)
                
                if skip_until_fit:  # If we never found model.compile, do manual fix
                    cell['source'] = new_lines
                    changes_made += 1
    
    # Save updated notebook
    if changes_made > 0:
        print(f"\n✓ Made {changes_made} changes. Saving notebook...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Notebook updated successfully!")
    else:
        print("\n⚠ No changes made. May need manual fix.")
    
    return changes_made

if __name__ == '__main__':
    notebook_path = 'submission_s3715228_s3343711_s4139514.ipynb'
    fix_notebook(notebook_path)
