import os
import sys
import numpy as np
import xgboost as xgb
import time

# Add python path
try:
    sys.path.insert(0, "/workspace/external/AarchGate/build/bindings/python")
    sys.path.insert(0, "/workspace")
    import aarchgate_python
    import bindings.python.aarchgate as apex
    from bindings.python.converter import XGBConverter
except ImportError:
    print("Failed to import aarchgate")
    sys.exit(1)

def train_dummy_model():
    print("Training XGBoost model...")
    import pandas as pd
    np.random.seed(42)
    X = np.random.normal(0, 1, (1000, 5))
    X = (X * 1000).astype(np.int64)
    df = pd.DataFrame(X, columns=["distance", "passenger_count", "pickup_hour", "day_of_week", "is_weekend"])
    y = np.random.normal(0, 1, 1000)
    y = (y * 1000).astype(np.int64)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=1,
        learning_rate=0.1,
        tree_method='hist'
    )
    model.fit(df, y)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/taxi_model.json")

def run_test():
    model_path = "models/taxi_model.json"
    if not os.path.exists(model_path):
        train_dummy_model()
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Create small data
    np.random.seed(42)
    X_small = np.random.normal(0, 1, (10000, 5))
    # Map to positive domain
    X_quantized = (X_small * 1000 + 1000.0).astype(np.uint64)
    
    stride = 40
    fields = [
        ("distance", 0, 64, 0),
        ("passenger_count", 8, 64, 0),
        ("pickup_hour", 16, 64, 0),
        ("day_of_week", 24, 64, 0),
        ("is_weekend", 32, 64, 0),
    ]
    
    converter = XGBConverter(precision_multiplier=1.0)
    model_json = converter.load_model(model_path)
    ir_root = converter.convert(model_json)
    
    engine = apex.ApexEngine()
    engine.register_schema("test_schema", fields, stride)
    engine.set_logic("test_schema", ir_root, 0)
    
    data_array = np.ascontiguousarray(X_quantized).view(np.uint8).ravel()
    
    # 1. Execute Row-Oriented
    total_sum_row = engine.execute(data_array, 10000)
    print(f"Row-oriented Sum: {total_sum_row}")
    
    # 2. Pre-slice
    def vectorize_all_pre_sliced(X, num_fields):
        num_blocks = X.shape[0] // 64
        out = np.zeros((num_blocks, num_fields, 64), dtype=np.uint64)
        shifts = np.arange(64, dtype=np.uint64)
        
        for b in range(num_blocks):
            block_X = X[b * 64 : (b + 1) * 64]
            for f in range(num_fields):
                cols = block_X[:, f]
                bits = (cols[:, None] >> shifts) & 1
                packed = np.sum(bits << shifts[:, None], axis=0)
                out[b, f, :] = packed
        return out.ravel()
        
    # We need count to be multiple of 64
    X_64 = X_quantized[:9984] # 156 blocks
    pre_sliced = vectorize_all_pre_sliced(X_64, 5)
    
    total_sum_row_64 = engine.execute(data_array[:9984*stride], 9984)
    print(f"Row-oriented Sum (9984 rows): {total_sum_row_64}")
    
    # 3. Direct Bitplane Comparison
    # Let's extract the first block, first field from python's pre_sliced
    py_block0_f0 = pre_sliced[:64]
    
    # Let's extract the first block, first field from C++ using our schema gathering & slicer
    # We can do this by running a simple test schema that loads f0 and we can inspect the bitplanes
    print("\n=== Slicing Comparison (Python vs C++) ===")
    print(f"Python first 5 words: {[hex(x) for x in py_block0_f0[:5]]}")
    
    # Let's check: what are the original 64 values of feature 0 (distance) in block 0?
    orig_vals = X_64[:64, 0]
    print(f"Original first 5 values: {list(orig_vals[:5])}")
    
    # Let's reconstruct the bitplanes manually:
    # Bit-plane p word is formed by: Sum_{r=0}^{63} (((orig_vals[r] >> p) & 1) << r)
    manual_block0_f0 = []
    for p in range(64):
        word = 0
        for r in range(64):
            bit = (int(orig_vals[r]) >> p) & 1
            word |= (bit << r)
        manual_block0_f0.append(word)
        
    print(f"Manual first 5 words: {[hex(x) for x in manual_block0_f0[:5]]}")
    
    # Let's check if Python pre_sliced matches manual reconstruction
    is_py_match = (list(py_block0_f0) == manual_block0_f0)
    print(f"Python matches Manual Reconstruction: {is_py_match}")
    if not is_py_match:
        for i in range(5):
            print(f"Index {i}: Py={hex(py_block0_f0[i])}, Manual={hex(manual_block0_f0[i])}")
            
    # 4. Block-by-block comparison
    print("\n=== Block-by-Block Comparison (156 Blocks) ===")
    mismatches = 0
    for b in range(156):
        # Row-oriented sum for this block
        block_data = data_array[b * 64 * stride : (b + 1) * 64 * stride]
        block_sum_row = engine.execute(block_data, 64)
        
        # Native sum for this block
        block_pre_sliced = pre_sliced[b * 5 * 64 : (b + 1) * 5 * 64]
        block_sum_native = engine.execute_native("test_schema", block_pre_sliced, 1, parallel=False)
        
        if block_sum_row != block_sum_native:
            mismatches += 1
            if mismatches <= 5:
                # Let's print the signed value
                row_val = int(block_sum_row - 18446744073709551616) if block_sum_row > 10**18 else int(block_sum_row)
                nat_val = int(block_sum_native - 18446744073709551616) if block_sum_native > 10**18 else int(block_sum_native)
                print(f"Block {b} Mismatch: Row={row_val} ({block_sum_row}), Native={nat_val} ({block_sum_native})")
                
    print(f"Total Block Mismatches: {mismatches} out of 156")
    
    sum_native = engine.execute_native("test_schema", pre_sliced, 156, parallel=False)
    print(f"Native Non-Parallel Sum: {sum_native}")
    
    sum_native_parallel = engine.execute_native("test_schema", pre_sliced, 156, parallel=True, num_threads=4)
    print(f"Native Parallel Sum: {sum_native_parallel}")

if __name__ == "__main__":
    run_test()
