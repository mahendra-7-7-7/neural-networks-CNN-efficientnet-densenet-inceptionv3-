"""
Pipeline Orchestrator for Pneumonia Detection
Runs InceptionV3 -> EfficientNetB3 -> Ensemble Evaluation
"""
import subprocess
import sys
import time

def run_step(step_name, cmd):
    print("\n" + "="*70)
    print(f"[*] STARTING: {step_name}")
    print(f"[*] Command: {' '.join(cmd)}")
    print("="*70 + "\n", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"\n[!] ERROR in {step_name}: Exited with code {res.returncode} (took {dt/60:.1f} min)")
        sys.exit(res.returncode)
    print(f"\n[✓] SUCCESS: {step_name} completed in {dt/60:.1f} minutes\n", flush=True)

if __name__ == '__main__':
    py = sys.executable
    print("="*70)
    print("CHEST X-RAY MULTI-MODEL PIPELINE (GPU)")
    print("="*70)
    
    # 1. InceptionV3
    run_step("InceptionV3 Training", [py, "train_inception.py"])
    
    # 2. EfficientNetB3
    run_step("EfficientNetB3 Training", [py, "train_efficientnet.py"])
    
    # 3. Ensemble Evaluation
    run_step("Ensemble Evaluation & Export", [py, "evaluate_ensemble.py"])
    
    print("\n" + "="*70)
    print("[✓] ALL MODELS AND ENSEMBLE EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
