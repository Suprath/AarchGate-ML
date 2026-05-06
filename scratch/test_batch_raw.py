import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from bindings.python.aarchgate_ml import AarchGateRegressor

def main():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=50, random_state=42)
    xgb_model.fit(X, y)
    
    model = AarchGateRegressor.from_xgboost(xgb_model)
    data_bytes, num_rows = model._prepare_data(X)
    stride_bytes = len(model.feature_names) * 8
    
    print(f"Calling execute_batch on {num_rows} rows...")
    t0 = time.time()
    res = model.engine.execute_batch(data_bytes, num_rows, model.precision_multiplier)
    dur = time.time() - t0
    print(f"Direct execute_batch duration: {dur:.6f}s for {num_rows} rows")
    print(f"Result type: {type(res)}, shape: {res.shape if hasattr(res, 'shape') else 'None'}, first 5: {res[:5]}")

if __name__ == "__main__":
    main()
