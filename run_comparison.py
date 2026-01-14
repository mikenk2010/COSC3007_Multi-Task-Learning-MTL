#!/usr/bin/env python3
"""
Run both notebooks and compare performance
WARNING: This will take 30-60 minutes to complete
"""

import subprocess
import sys
import time

def run_notebook(notebook_name, timeout=3600):
    """Execute a notebook and track time"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {notebook_name}")
    print(f"Timeout: {timeout}s ({timeout//60} minutes)")
    print(f"{'='*70}\n")
    
    output_name = notebook_name.replace('.ipynb', '_RUN.ipynb')
    
    start = time.time()
    
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            f'--ExecutePreprocessor.timeout={timeout}',
            '--output', output_name,
            notebook_name
        ], capture_output=True, text=True, timeout=timeout+60)
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ SUCCESS in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"✓ Output saved to: {output_name}\n")
            return True, elapsed, output_name
        else:
            print(f"✗ FAILED after {elapsed:.1f}s")
            print(f"Error: {result.stderr}\n")
            return False, elapsed, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT after {timeout}s\n")
        return False, timeout, None
    except Exception as e:
        print(f"✗ ERROR: {e}\n")
        return False, 0, None

def main():
    print("\n" + "="*70)
    print("DEEP LEARNING COURSE - NOTEBOOK COMPARISON")
    print("="*70)
    print("\nThis will:")
    print("  1. Run test_clean.ipynb (Simple CNN) - ~15 min")
    print("  2. Run submission_xxxx_testclean.ipynb (ResNet) - ~30 min")
    print("  3. Generate comparison report")
    print("\nTotal estimated time: 45-60 minutes")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)
    
    results = {}
    
    # Experiment 1: Simple CNN
    success1, time1, out1 = run_notebook('test_clean.ipynb', timeout=1800)
    results['test_clean'] = {
        'success': success1,
        'time': time1,
        'output': out1
    }
    
    # Experiment 2: ResNet
    success2, time2, out2 = run_notebook('submission_xxxx_testclean.ipynb', timeout=3600)
    results['submission'] = {
        'success': success2,
        'time': time2,
        'output': out2
    }
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"\ntest_clean.ipynb:")
    print(f"  Status: {'✓ Success' if success1 else '✗ Failed'}")
    print(f"  Time: {time1:.1f}s ({time1/60:.1f} min)")
    if out1:
        print(f"  Output: {out1}")
    
    print(f"\nsubmission_xxxx_testclean.ipynb:")
    print(f"  Status: {'✓ Success' if success2 else '✗ Failed'}")
    print(f"  Time: {time2:.1f}s ({time2/60:.1f} min)")
    if out2:
        print(f"  Output: {out2}")
    
    if success1 and success2:
        print(f"\n✓ Both notebooks executed successfully!")
        print(f"✓ Check the *_RUN.ipynb files for detailed results")
        print(f"\nNext: Open the notebooks to compare:")
        print(f"  jupyter notebook {out1}")
        print(f"  jupyter notebook {out2}")
    else:
        print(f"\n⚠ Some notebooks failed to execute")
        print(f"Check error messages above")

if __name__ == '__main__':
    main()
