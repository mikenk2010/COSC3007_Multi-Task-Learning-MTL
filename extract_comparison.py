#!/usr/bin/env python3
"""
Extract and compare existing results from both notebooks
"""

import json
import re

def extract_metrics_from_notebook(notebook_path):
    """Extract all performance metrics from executed notebook"""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    metrics = {
        'model_name': None,
        'parameters': None,
        'training_time': None,
        'epochs': None,
        'target_a_val_acc': None,
        'target_b_val_acc': None,
        'target_c_val_mae': None,
        'target_a_test_acc': None,
        'target_b_test_acc': None,
        'target_c_test_mae': None,
        'best_val_a': None,
        'best_val_b': None,
        'best_val_c': None
    }
    
    all_text = []
    
    # Collect all outputs
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = ''.join(output['text'])
                    all_text.append(text)
    
    # Join all text for searching
    full_output = '\n'.join(all_text)
    
    # Extract metrics using patterns
    patterns = {
        'target_a_test_acc': r'Target A.*?Accuracy:\s*([\d.]+)%',
        'target_b_test_acc': r'Target B.*?Accuracy:\s*([\d.]+)%',
        'target_c_test_mae': r'Target C.*?MAE:\s*([\d.]+)',
        'best_val_a': r'Best validation.*?Target A:\s*([\d.]+)%',
        'best_val_b': r'Best validation.*?Target B:\s*([\d.]+)%',
        'best_val_c': r'Best validation.*?Target C:\s*([\d.]+)',
        'epochs': r'Training completed in (\d+) epochs',
        'baseline_acc': r'Baseline.*?Validation Accuracy:\s*([\d.]+)%'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, full_output, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                metrics[key] = match.group(1)
    
    return metrics

def print_comparison(metrics1, metrics2, name1, name2):
    """Print side-by-side comparison"""
    
    print("\n" + "="*90)
    print(f"{'PERFORMANCE COMPARISON':^90}")
    print("="*90)
    print(f"\n{'Metric':<40} {name1:<25} {name2:<25}")
    print("-" * 90)
    
    comparisons = [
        ("Final Test - Target A Accuracy (%)", 'target_a_test_acc', 'higher'),
        ("Final Test - Target B Accuracy (%)", 'target_b_test_acc', 'higher'),
        ("Final Test - Target C MAE", 'target_c_test_mae', 'lower'),
        ("", None, None),  # Separator
        ("Best Validation - Target A (%)", 'best_val_a', 'higher'),
        ("Best Validation - Target B (%)", 'best_val_b', 'higher'),
        ("Best Validation - Target C MAE", 'best_val_c', 'lower'),
        ("", None, None),  # Separator
        ("Training Epochs", 'epochs', 'lower'),
        ("Baseline Accuracy (%)", 'baseline_acc', 'higher'),
    ]
    
    for metric_name, key, better in comparisons:
        if key is None:
            print()
            continue
            
        val1 = metrics1.get(key, 'N/A')
        val2 = metrics2.get(key, 'N/A')
        
        # Format values
        val1_str = f"{val1:.2f}" if isinstance(val1, float) else str(val1)
        val2_str = f"{val2:.2f}" if isinstance(val2, float) else str(val2)
        
        # Determine winner
        winner = ""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if better == 'higher':
                if val1 > val2:
                    val1_str = f"{val1_str} ✓"
                elif val2 > val1:
                    val2_str = f"{val2_str} ✓"
            elif better == 'lower':
                if val1 < val2:
                    val1_str = f"{val1_str} ✓"
                elif val2 < val1:
                    val2_str = f"{val2_str} ✓"
        
        print(f"{metric_name:<40} {val1_str:<25} {val2_str:<25}")
    
    print("\n" + "="*90)

def main():
    print("\n" + "="*90)
    print(f"{'DEEP LEARNING COURSE - NOTEBOOK RESULTS COMPARISON':^90}")
    print("="*90)
    
    # Extract metrics
    print("\nExtracting metrics from test_clean.ipynb...")
    metrics1 = extract_metrics_from_notebook('test_clean.ipynb')
    
    print("Extracting metrics from submission_xxxx_testclean.ipynb...")
    metrics2 = extract_metrics_from_notebook('submission_xxxx_testclean.ipynb')
    
    # Print comparison
    print_comparison(
        metrics1, metrics2,
        "test_clean.ipynb", 
        "submission.ipynb"
    )
    
    # Additional analysis
    print("\n" + "="*90)
    print("KEY FINDINGS")
    print("="*90)
    
    if metrics1.get('target_b_test_acc') and metrics2.get('target_b_test_acc'):
        b1 = metrics1['target_b_test_acc']
        b2 = metrics2['target_b_test_acc']
        
        print(f"\n1. Target B (32-class - The Difficult Task):")
        print(f"   test_clean.ipynb: {b1:.2f}%")
        print(f"   submission.ipynb: {b2:.2f}%")
        
        if b1 > b2:
            improvement = ((b1 - b2) / b2) * 100
            print(f"   → test_clean.ipynb is {improvement:.1f}% better (SIMPLER MODEL WINS!)")
        else:
            improvement = ((b2 - b1) / b1) * 100
            print(f"   → submission.ipynb is {improvement:.1f}% better")
    
    if metrics1.get('epochs') and metrics2.get('epochs'):
        print(f"\n2. Training Efficiency:")
        print(f"   test_clean.ipynb: {metrics1['epochs']} epochs")
        print(f"   submission.ipynb: {metrics2['epochs']} epochs")
    
    print(f"\n3. Architecture Complexity:")
    print(f"   test_clean.ipynb: Simple CNN (~200K params)")
    print(f"   submission.ipynb: ResNet-V2 (~500K params)")
    
    print("\n" + "="*90)
    
    # Save to file
    with open('comparison_results.json', 'w') as f:
        json.dump({
            'test_clean': metrics1,
            'submission': metrics2
        }, f, indent=2)
    
    print("\n✓ Detailed results saved to: comparison_results.json")
    print("="*90 + "\n")

if __name__ == '__main__':
    main()
