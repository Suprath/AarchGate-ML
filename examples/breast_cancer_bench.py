import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from bindings.python.aarchgate import ApexEngine
from bindings.python.converter import XGBConverter

def main():
    print("=== Loading Real-World Breast Cancer Diagnosis Dataset (30 Features) ===")
    data = load_breast_cancer()
    # Use clean feature names "f0", "f1", ..., "f29" to avoid special characters
    feature_names = [f"f{i}" for i in range(30)]
    X = pd.DataFrame(data.data, columns=feature_names)
    y = data.target
    
    print(f"Dataset Loaded. Shape: {X.shape} (30 physical diagnostic features)")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost Classifier...")
    # Train model (max_depth=3, 50 trees to fit within register limits)
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    # Save model to JSON
    model_json_path = "models/breast_cancer.json"
    model.save_model(model_json_path)
    print(f"XGBoost Model saved to {model_json_path}")
    
    print("\nConverting model to AarchGate circuit format...")
    # Scale continuous floats by 10^5 to keep precise threshold partitions
    scale_factor = 100000.0
    converter = XGBConverter(precision_multiplier=scale_factor)
    model_json = converter.load_model(model_json_path)
    ir_root = converter.convert(model_json)
    
    # Initialize Engine
    engine = ApexEngine()
    
    # Register schema: 30 fields, offset is index * 8, 64 bits per field
    fields = [(f"f{i}", i * 8, 64, 0) for i in range(30)]
    stride_bytes = 30 * 8  # 240 bytes per row
    engine.register_schema("cancer_schema", fields, stride_bytes)
    engine.set_logic("cancer_schema", ir_root, 2)
    
    print("Preparing quantized dataset...")
    num_test_rows = X_test.shape[0]
    
    # Format quantized data as a continuous 1D uint64 array
    # Shape: num_test_rows * 30
    X_test_quantized = np.zeros(num_test_rows * 30, dtype=np.uint64)
    for r in range(num_test_rows):
        row_vals = X_test.iloc[r].values
        for f in range(30):
            # Scale and shift by +1000.0 to prevent negative values
            X_test_quantized[r * 30 + f] = int(round((row_vals[f] + 1000.0) * scale_factor))
            
    # Verify and compare Predictions
    # XGBoost outputs log odds leaf predictions that are summed
    # We obtain raw leaf predictions (log-odds margins) from XGBoost booster to compare directly
    booster = model.get_booster()
    # predict(output_margin=True) returns raw leaf summations before sigmoid
    xgb_margins = booster.predict(xgb.DMatrix(X_test), output_margin=True)
    # Scale to match the precision multiplier
    xgb_margins_scaled_sum = int(np.sum(xgb_margins) * scale_factor)
    
    # Run dynamic AarchGate on-the-fly row-by-row
    data_array = np.ascontiguousarray(X_test_quantized).view(np.uint8).ravel()
    
    print("\n=== First 10 Rows Comparison ===")
    print(f"{'Row':<5} | {'XGBoost Margin':<15} | {'AarchGate Margin':<15} | {'Match'}")
    print("-" * 50)
    for r in range(10):
        row_view = data_array[r * stride_bytes : (r + 1) * stride_bytes]
        ag_val_raw = engine.execute(row_view, 1)
        ag_val = ag_val_raw if ag_val_raw < 2**63 else ag_val_raw - 2**64
        xgb_scaled = int(xgb_margins[r] * scale_factor)
        # Check if they are close (allowing small rounding tolerance)
        is_match = abs(ag_val - xgb_scaled) <= 5
        print(f"{r:<5} | {xgb_scaled:<15} | {ag_val:<15} | {is_match}")

    aarchgate_sum_raw = engine.execute(data_array, num_test_rows, parallel=False)
    aarchgate_sum = aarchgate_sum_raw if aarchgate_sum_raw < 2**63 else aarchgate_sum_raw - 2**64
    
    aarchgate_parallel_raw = engine.execute(data_array, num_test_rows, parallel=True, num_threads=4)
    aarchgate_parallel_sum = aarchgate_parallel_raw if aarchgate_parallel_raw < 2**63 else aarchgate_parallel_raw - 2**64
    
    print("\n=== Real-World Verification ===")
    print(f"XGBoost Total Cumulative Margin Sum: {xgb_margins_scaled_sum}")
    print(f"AarchGate Dynamic Sequential Sum:     {aarchgate_sum}")
    print(f"AarchGate Dynamic Parallel Sum:       {aarchgate_parallel_sum}")
    
    # Compute matching accuracy
    margin = abs(xgb_margins_scaled_sum - aarchgate_sum) / abs(xgb_margins_scaled_sum) * 100
    print(f"Parity Match: {100.0 - margin:.4f}% exact match")
    
    print("\n=== Running Latency & Throughput Benchmark ===")
    # Replicate test dataset to 1M samples for stable high-throughput benchmark
    benchmark_multiplier = 10000
    X_bench_quantized = np.tile(X_test_quantized, benchmark_multiplier)
    bench_rows = num_test_rows * benchmark_multiplier
    
    # Convert to raw 1D byte array for AarchGate ingestion
    bench_data_bytes = np.ascontiguousarray(X_bench_quantized).view(np.uint8).ravel()
    
    print(f"Evaluating parallel inference on {bench_rows} rows using 4 threads...")
    
    start_time = time.time()
    total_predictions_sum_raw = engine.execute(bench_data_bytes, bench_rows, parallel=True, num_threads=4)
    total_predictions_sum = total_predictions_sum_raw if total_predictions_sum_raw < 2**63 else total_predictions_sum_raw - 2**64
    duration = time.time() - start_time
    
    throughput = bench_rows / duration
    print(f"Completed in {duration:.4f} seconds.")
    print(f"Throughput: {throughput / 1e6:.2f} Million rows per second!")
    print(f"Average Latency: {duration / bench_rows * 1e9:.2f} nanoseconds per row!")
    print(f"Aggregate Prediction Sum: {total_predictions_sum}")

if __name__ == "__main__":
    main()
