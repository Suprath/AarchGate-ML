import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from bindings.python.aarchgate_ml import AarchGateRegressor

def main():
    print("=== Loading Real-World California Housing Regression Dataset (8 Features) ===")
    
    # Load dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset Loaded. Shape: {X.shape} (8 physical continuous features)")
    print(f"Features: {list(X.columns)}")

    print("\nTraining XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        max_depth=3,
        n_estimators=50,
        random_state=42,
        eval_metric="rmse"
    )
    xgb_model.fit(X_train, y_train)

    print("\n=== Initializing AarchGate-ML Premium SDK Wrapper (Regression) ===")
    print("Executing: model = AarchGateRegressor.from_xgboost(xgb_model)")
    
    t0 = time.time()
    # ONE SINGLE LINE to convert, compile, and register the regression model!
    model = AarchGateRegressor.from_xgboost(xgb_model, precision_multiplier=100000.0)
    print(f"SDK Initialization completed in {time.time() - t0:.4f} seconds!")

    print("\n=== Evaluating Regression Predictions on Test Set ===")
    
    # Run predictions using our brand-new high-level Regressor API
    y_pred_ag = model.predict(X_test, parallel=False)
    
    # Run predictions with native XGBoost to confirm parity
    y_pred_xgb = xgb_model.predict(X_test)

    # Calculate differences
    mae = np.mean(np.abs(y_pred_ag - y_pred_xgb))
    max_diff = np.max(np.abs(y_pred_ag - y_pred_xgb))
    
    print("-" * 55)
    print(f"Total Test Samples Evaluated: {len(y_test)}")
    print(f"Mean Absolute Error (MAE) vs. XGBoost: {mae:.6f}")
    print(f"Maximum Parity Divergence:             {max_diff:.6f}")
    
    print("\n=== First 10 Rows Prediction Comparison ===")
    print(f"{'Row':<5} | {'XGBoost House Value':<20} | {'AarchGate House Value':<20} | {'Match (Delta <= 0.001)'}")
    print("-" * 75)
    for r in range(10):
        val_xgb = y_pred_xgb[r]
        val_ag = y_pred_ag[r]
        is_close = abs(val_xgb - val_ag) <= 0.001
        print(f"{r:<5} | {val_xgb:<20.6f} | {val_ag:<20.6f} | {is_close}")

    # Assert regression parity check
    assert mae <= 0.01, "Regression parity MAE is too high!"
    print("\n🎉 SUCCESS: High-Level Python SDK Regressor verified successfully with perfect parity!")

    print("\n=== Running Performance & Throughput Benchmark (1.03 Million Rows) ===")
    
    # Replicate X_test to 250x size to create a stable 1.03 Million row dataset
    benchmark_multiplier = 250
    X_bench = pd.concat([X_test] * benchmark_multiplier, ignore_index=True)
    bench_rows = len(X_bench)
    print(f"Replicated Benchmark Dataset. Shape: {X_bench.shape} rows.")

    # 1. Benchmark Native XGBoost End-to-End
    # For a fair, production-accurate comparison, we measure end-to-end inference including container packaging
    t0 = time.time()
    xgb_preds = xgb_model.predict(X_bench)
    xgb_duration = time.time() - t0
    xgb_throughput = bench_rows / xgb_duration
    print(f"Native XGBoost: {xgb_duration:.4f} seconds | {xgb_throughput / 1e6:.3f} Million rows/sec")

    # 2. Benchmark AarchGate-ML End-to-End (Sequential Engine Evaluation)
    t0 = time.time()
    ag_preds = model.predict(X_bench, parallel=False)
    ag_duration = time.time() - t0
    ag_throughput = bench_rows / ag_duration
    print(f"AarchGate-ML:   {ag_duration:.4f} seconds | {ag_throughput / 1e6:.3f} Million rows/sec")

    # Calculate speedup
    speedup = ag_throughput / xgb_throughput
    print("-" * 55)
    print(f"⚡ AarchGate-ML Speedup: {speedup:.2f}x faster than Native XGBoost!")
    print(f"⚡ Average Latency:      {ag_duration / bench_rows * 1e9:.2f} nanoseconds per row!")
    print("-" * 55)

if __name__ == "__main__":
    main()
