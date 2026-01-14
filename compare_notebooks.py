#!/usr/bin/env python3
"""
Compare performance of test_clean.ipynb vs submission_xxxx_testclean.ipynb
"""

import subprocess
import json
import time
import os
import sys

def run_notebook(notebook_path, timeout=1800):
    """
    Execute a Jupyter notebook and return execution time.
    
    Args:
        notebook_path: Path to .ipynb file
        timeout: Maximum execution time in seconds (default: 30 min)
    
    Returns:
        dict with execution_time, success, and output_path
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {notebook_path}")
    print(f"{'='*60}\n")
    
    output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    
    start_time = time.time()
    
    try:
        # Run notebook using jupyter nbconvert
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=1800',  # 30 min timeout
            '--output', output_path,
            notebook_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Successfully executed in {execution_time:.1f}s")
            return {
                'success': True,
                'execution_time': execution_time,
                'output_path': output_path,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"✗ Execution failed!")
            print(f"STDERR: {result.stderr}")
            return {
                'success': False,
                'execution_time': execution_time,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"✗ Timeout after {execution_time:.1f}s")
        return {
            'success': False,
            'execution_time': execution_time,
            'error': 'Timeout'
        }
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"✗ Error: {e}")
        return {
            'success': False,
            'execution_time': execution_time,
            'error': str(e)
        }

def extract_metrics(notebook_path):
    """
    Extract key performance metrics from executed notebook.
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        metrics = {}  # Use dict instead of typed dict to avoid type errors
        
        # Search through cells for metrics
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        text = ''.join(output['text'])
                        
                        # Extract Target A accuracy
                        if 'Target A' in text and 'Accuracy:' in text:
                            for line in text.split('\n'):
                                if 'Accuracy:' in line and '%' in line:
                                    try:
                                        acc = float(line.split(':')[1].strip().rstrip('%'))
                                        metrics['target_a_accuracy'] = acc
                                    except:
                                        pass
                        
                        # Extract Target B accuracy
                        if 'Target B' in text and 'Accuracy:' in text:
                            for line in text.split('\n'):
                                if 'Accuracy:' in line and '%' in line:
                                    try:
                                        acc = float(line.split(':')[1].strip().rstrip('%'))
                                        metrics['target_b_accuracy'] = acc
                                    except:
                                        pass
                        
                        # Extract Target C MAE
                        if 'Target C' in text and 'MAE:' in text:
                            for line in text.split('\n'):
                                if 'MAE:' in line:
                                    try:
                                        mae = float(line.split(':')[1].strip())
                                        metrics['target_c_mae'] = mae
                                    except:
                                        pass
                        
                        # Extract best validation metrics
                        if 'Best validation' in text:
                            for line in text.split('\n'):
                                if 'Target A:' in line and '%' in line:
                                    try:
                                        acc = float(line.split(':')[1].strip().rstrip('%'))
                                        metrics['best_val_accuracy_a'] = acc
                                    except:
                                        pass
                                if 'Target B:' in line and '%' in line:
                                    try:
                                        acc = float(line.split(':')[1].strip().rstrip('%'))
                                        metrics['best_val_accuracy_b'] = acc
                                    except:
                                        pass
                                if 'Target C:' in line and 'MAE' in line:
                                    try:
                                        mae = float(line.split(':')[1].strip().split()[0])
                                        metrics['best_val_mae_c'] = mae
                                    except:
                                        pass
        
        return metrics
        
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return None

def main():
    # Check if dataset exists
    if not os.path.exists('dataset_dev_3000.npz'):
        print("ERROR: dataset_dev_3000.npz not found!")
        sys.exit(1)
    
    results = {}
    
    # Run test_clean.ipynb
    print("\n" + "="*60)
    print("EXPERIMENT 1: test_clean.ipynb (Simple CNN)")
    print("="*60)
    
    result1 = run_notebook('test_clean.ipynb', timeout=1800)
    results['test_clean'] = result1
    
    if result1['success']:
        metrics1 = extract_metrics(result1['output_path'])
        results['test_clean']['metrics'] = metrics1
    
    # Run submission notebook
    print("\n" + "="*60)
    print("EXPERIMENT 2: submission_xxxx_testclean.ipynb (ResNet-V2)")
    print("="*60)
    
    result2 = run_notebook('submission_xxxx_testclean.ipynb', timeout=3600)  # 1 hour
    results['submission'] = result2
    
    if result2['success']:
        metrics2 = extract_metrics(result2['output_path'])
        results['submission']['metrics'] = metrics2
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'test_clean.ipynb':<20} {'submission.ipynb':<20}")
    print("-" * 70)
    
    if results['test_clean']['success'] and results['submission']['success']:
        m1 = results['test_clean'].get('metrics', {})
        m2 = results['submission'].get('metrics', {})
        
        print(f"{'Execution Time (s)':<30} {results['test_clean']['execution_time']:<20.1f} {results['submission']['execution_time']:<20.1f}")
        print(f"{'Target A Accuracy (%)':<30} {str(m1.get('target_a_accuracy', 'N/A')):<20} {str(m2.get('target_a_accuracy', 'N/A')):<20}")
        print(f"{'Target B Accuracy (%)':<30} {str(m1.get('target_b_accuracy', 'N/A')):<20} {str(m2.get('target_b_accuracy', 'N/A')):<20}")
        print(f"{'Target C MAE':<30} {str(m1.get('target_c_mae', 'N/A')):<20} {str(m2.get('target_c_mae', 'N/A')):<20}")
        print(f"{'Best Val Acc A (%)':<30} {str(m1.get('best_val_accuracy_a', 'N/A')):<20} {str(m2.get('best_val_accuracy_a', 'N/A')):<20}")
        print(f"{'Best Val Acc B (%)':<30} {str(m1.get('best_val_accuracy_b', 'N/A')):<20} {str(m2.get('best_val_accuracy_b', 'N/A')):<20}")
        print(f"{'Best Val MAE C':<30} {str(m1.get('best_val_mae_c', 'N/A')):<20} {str(m2.get('best_val_mae_c', 'N/A')):<20}")
    
    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Full results saved to comparison_results.json")

if __name__ == '__main__':
    main()
