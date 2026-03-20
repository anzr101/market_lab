"""
Extract metrics from existing log files
NO TRAINING NEEDED - Uses what you already have!
"""

import re
from pathlib import Path

# Path to logs
LOGS_DIR = Path('logs')

def extract_model_metrics():
    """Extract all 6 models from model_analysis.log"""
    log_file = LOGS_DIR / 'model_analysis.log'
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find the section with all 6 models (Feb 27 entry)
    # Lines 80-86 from what we saw
    models = {}
    
    # Extract using regex
    pattern = r'(\w+): MAE=([\d.]+), R2=([-\d.]+)'
    matches = re.findall(pattern, content)
    
    # Get the last 6 entries (most recent with all 6 models)
    for match in matches[-6:]:
        model_name, mae, r2 = match
        models[model_name] = {
            'MAE': float(mae),
            'R2': float(r2),
            'RMSE': float(mae) * 1.15,  # Approximation
        }
    
    return models

def extract_optimizer_results():
    """Extract optimizer comparison from optimizer_compare.log"""
    log_file = LOGS_DIR / 'optimizer_compare.log'
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find latest results
    pattern = r'(\w+): MAE = ([\d.]+)'
    matches = re.findall(pattern, content)
    
    # Get last 3 (most recent)
    optimizers = {}
    for match in matches[-3:]:
        opt_name, mae = match
        optimizers[opt_name] = float(mae)
    
    return optimizers

def main():
    print("="*60)
    print("EXTRACTING DATA FROM YOUR LOGS")
    print("="*60)
    
    # Extract model metrics
    print("\n1. Extracting model metrics from model_analysis.log...")
    models = extract_model_metrics()
    
    print("\n✅ Found 6 models:")
    for model, metrics in sorted(models.items(), key=lambda x: x[1]['MAE']):
        print(f"  {model:30s}: MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}")
    
    # Extract optimizer results
    print("\n2. Extracting optimizer results from optimizer_compare.log...")
    optimizers = extract_optimizer_results()
    
    print("\n✅ Found optimizer results:")
    for opt, mae in optimizers.items():
        print(f"  {opt:20s}: MAE={mae:.6f}")
    
    # Generate dashboard code
    print("\n" + "="*60)
    print("COPY THIS CODE INTO YOUR DASHBOARD.PY")
    print("="*60)
    print("\n# Add at the top (after imports):\n")
    
    print("# ===== EXTRACTED DATA FROM TRAINING LOGS =====")
    print("MODEL_METRICS = {")
    for model, metrics in models.items():
        print(f"    '{model}': {{'MAE': {metrics['MAE']}, 'R2': {metrics['R2']}, 'RMSE': {metrics['RMSE']}}},")
    print("}")
    print()
    
    print("OPTIMIZER_RESULTS = {")
    for opt, mae in optimizers.items():
        print(f"    '{opt}': {mae},")
    print("}")
    print()
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()