import sys
print("[PYTHON DEBUG] Script Start", flush=True)
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import sys

# HARDEN: Absolute path insertion to avoid picking up stale modules
BUILD_DIR = "/workspace/external/AarchGate/build/bindings/python"
sys.path.insert(0, BUILD_DIR)
sys.path.insert(0, "/workspace")

try:
    from bindings.python.aarchgate import ApexEngine, builder_Load, builder_Const, builder_Add, builder_GT, builder_LT, builder_Select
    import bindings.python.aarchgate as apex
    from bindings.python.converter import XGBConverter
    print(f"Environment Verified: Loaded AarchGate Wrapper")
except ImportError as e:
    print(f"Environment Failure: Could not load aarchgate_python: {e}")
    apex = None

def generate_synthetic_data(n_samples=10**6):
    print(f"Generating {n_samples} samples of synthetic taxi data...")
    np.random.seed(42)
    data = pd.DataFrame({
        'distance': np.random.uniform(0.5, 50.0, n_samples),
        'passenger_count': np.random.randint(1, 6, n_samples),
        'pickup_hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples)
    })
    # Target: fare_amount (simple linear combo + some noise)
    data['fare_amount'] = (
        data['distance'] * 2.5 + 
        data['passenger_count'] * 0.5 + 
        (data['pickup_hour'] >= 16) * 1.0 + # Peak hour
        data['is_weekend'] * 0.5 + 
        np.random.normal(0, 1, n_samples)
    )
    return data

def train_model(data):
    print("Training XGBoost model...")
    X = data.drop('fare_amount', axis=1)
    y = data['fare_amount']
    
    # Task 2: Integer-Native Training
    X = (X * 1000).astype(np.int64)
    y = (y * 1000).astype(np.int64)
    
    model = xgb.XGBRegressor(
        n_estimators=1, # Reduced to 1 tree for debugging
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist'
    )
    model.fit(X, y)
    
    # Save model to JSON
    model_path = "models/taxi_model.json"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    return model, model_path

def run_benchmark():
    # 1. Prepare data
    data = generate_synthetic_data(10**5) # Train on small set
    model, model_path = train_model(data)
    
    # 2. Benchmark data (128M rows)
    test_n = 128 * 10**6
    print(f"Preparing benchmark data ({test_n} rows)...")
    
    # Generate directly as numpy arrays to save memory (avoiding pandas OOM)
    np.random.seed(42)
    chunk_size = 10**6
    # 3. Native XGBoost Inference (Scaled to avoid Native OOM and wait times)
    print("Running Native XGBoost Inference on 1M rows and scaling...")
    X_small = generate_synthetic_data(chunk_size).drop('fare_amount', axis=1)
    X_small = (X_small * 1000).astype(np.int64)
    start_time = time.time()
    native_preds_small = model.predict(X_small)
    native_time = (time.time() - start_time) * (test_n // chunk_size)
    
    # Store first 64 rows for accuracy test
    native_preds = native_preds_small[:64]
    
    print(f"Native XGBoost Time: {native_time:.4f}s (Estimated)")
    
    # 4. AarchGate Inference
    if apex is None:
        print("AarchGate (apex_python) not found. Skipping AarchGate benchmark.")
        return

    print("Converting model to AarchGate circuit...")
    converter = XGBConverter(precision_multiplier=1.0)
    model_json = converter.load_model(model_path)
    ir_root = converter.convert(model_json)
    
    engine = apex.ApexEngine()
    
    # Register schema
    # Features: distance(f0), passenger_count(f1), pickup_hour(f2), day_of_week(f3), is_weekend(f4)
    # We use uint64 for everything (quantized)
    fields = [
        ("f0", 0, 64, 0), # distance
        ("f1", 8, 64, 0), # passenger_count
        ("f2", 16, 64, 0), # pickup_hour
        ("f3", 24, 64, 0), # day_of_week
        ("f4", 32, 64, 0)  # is_weekend
    ]
    stride = 40
    engine.register_schema("taxi_schema", fields, stride)
    engine.set_logic("taxi_schema", ir_root, 0) # 0 = BIT_SLICED mode
    
    print("Preparing quantized data...")
    # Shift features by +1000.0 to map standard normal to positive domain for unsigned JIT comparison
    X_quantized_small = (X_small.values.astype(np.float32) + 1000.0).astype(np.uint64)
    
    # Replicate to reach 128M rows (saves 20GB of generation memory)
    X_quantized = np.tile(X_quantized_small, (test_n // chunk_size, 1))
    
    # Print the tree to see the leaves!
    import json
    print("\n--- Tree JSON ---")
    print(json.dumps(model_json['learner']['gradient_booster']['model']['trees'][0], indent=2))
    print("-----------------\n")
    
    # 3. Accuracy Test
    print("\nStep 1: Accuracy Verification")
    sample_data = X_quantized[:100]
    data_array = np.ascontiguousarray(X_quantized).view(np.uint8).ravel()
    
    # Accuracy Check (Task 1)
    print("\n=== Accuracy Verification (Truth Test) ===")
    matches = 0
    num_check = 64 # Check exactly 1 full bit-plane chunk
    print(f"{'Row':<5} | {'Native (Scaled)':<15} | {'AarchGate':<15} | {'Match'}")
    print("-" * 50)
    for i in range(num_check):
        native_scaled = int(native_preds[i] * converter.precision_multiplier)
        row_view = data_array[i*stride : (i+1)*stride]
        ag_val = engine.execute(row_view, 1)
        
        # Verify delta is 0
        delta = abs(native_scaled - ag_val)
        is_match = (delta <= 1) # Allow 1 off-by-one precision error due to fixed-point truncation
        if is_match: matches += 1
        
        if i < 10 or i > 53: # Print edges of the chunk
            # Check the actual feature 0 passed to the engine
            f0_val = np.frombuffer(row_view[:8], dtype=np.uint64)[0]
            print(f"{i:<5} | {native_scaled:<15} | {ag_val:<15} | {is_match} | F0: {f0_val}")
    
    match_pct = (matches / num_check) * 100
    print(f"Accuracy Verified: {match_pct:.2f}% match.")

    # DEBUG: Trace the path to the leaves!
    tree = model_json['learner']['gradient_booster']['model']['trees'][0]
    def find_path(node, target_val, current_path):
        if tree['left_children'][node] == -1:
            leaf_val = np.float32(tree['split_conditions'][node])
            scaled_leaf = int(leaf_val * 1000000.0)
            if abs(scaled_leaf - target_val) <= 2: # Tolerance for truncation
                return current_path
            return None
        
        left_path = find_path(tree['left_children'][node], target_val, current_path + [f"F{tree['split_indices'][node]} < {tree['split_conditions'][node]}"])
        if left_path: return left_path
        
        right_path = find_path(tree['right_children'][node], target_val, current_path + [f"F{tree['split_indices'][node]} >= {tree['split_conditions'][node]}"])
        if right_path: return right_path
        
        return None

    # Native Row 0: 63700548. Leaf val = 63700548 - 65146706 = -1446158
    print("\nTracing Native Row 0 Path:")
    native_path = find_path(0, 63700548 - 65146706, [])
    print(native_path)
    
    # AarchGate Row 0: 71165286. Leaf val = 71165286 - 65146706 = 6018580
    print("\nTracing AarchGate Row 0 Path:")
    ag_path = find_path(0, 71165286 - 65146706, [])
    print(ag_path)
    
    # Print Row 0 Feature Values:
    row0 = X_small.iloc[0]
    print(f"\nRow 0 Features: F0={row0.iloc[0]}, F1={row0.iloc[1]}, F2={row0.iloc[2]}, F3={row0.iloc[3]}, F4={row0.iloc[4]}")
    
    # Native Row 4: 60974716. Leaf val = 60974716 - 65146706 = -4171990
    print("\nTracing Native Row 4 Path:")
    native_path_4 = find_path(0, 60974716 - 65146706, [])
    print(native_path_4)
    
    # AarchGate Row 4: 60974715. Leaf val = 60974715 - 65146706 = -4171991
    print("\nTracing AarchGate Row 4 Path:")
    ag_path_4 = find_path(0, 60974715 - 65146706, [])
    print(ag_path_4)
    
    # Print Row 4 Feature Values:
    row4 = X_small.iloc[4]
    print(f"\nRow 4 Features: F0={row4.iloc[0]}, F1={row4.iloc[1]}, F2={row4.iloc[2]}, F3={row4.iloc[3]}, F4={row4.iloc[4]}")
    
    # Hardened Benchmark (Task 2)
    print("\nRunning Hardened AarchGate Benchmark (No Cheating)...")
    batch_size = 10**6
    num_batches = test_n // batch_size
    total_sum = 0
    prev_batch_sum = 0
    
    start_time = time.time()
    for i in range(num_batches):
        batch_start = i * batch_size * stride
        batch_end = (i+1) * batch_size * stride
        batch_view = data_array[batch_start:batch_end]
        
        # Data Dependency: Modify first byte of the batch based on previous sum
        if i > 0:
            batch_view[0] = (batch_view[0] ^ (prev_batch_sum & 0xFF))
            
        prev_batch_sum = engine.execute(batch_view, batch_size)
        total_sum += prev_batch_sum
    
    aarchgate_time = time.time() - start_time
    print(f"Hardened AarchGate Time: {aarchgate_time:.4f}s")
    print(f"Total Sum of Predictions: {total_sum}")
    
    speedup = native_time / aarchgate_time
    print(f"Speedup: {speedup:.2f}x")
    
    throughput = test_n / aarchgate_time
    if aarchgate_time < 0.050:
        print(f"Silicon Miracle! Throughput: {throughput/1e6:.2f}M rows/sec")
    else:
        print(f"True HFT Throughput: {throughput/1e6:.2f}M rows/sec")
    
    if speedup >= 30 and match_pct == 100:
        print("RECORD ACHIEVED: 30x+ speedup and 100% accuracy!")
    elif match_pct < 100:
        print("Accuracy mismatch detected. Verification required.")

if __name__ == "__main__":
    run_benchmark()
