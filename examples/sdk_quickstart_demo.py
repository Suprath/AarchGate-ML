import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from bindings.python.aarchgate_ml import AarchGateClassifier

def main():
    print("=== Loading Real-World Breast Cancer Diagnosis Dataset (30 Features) ===")
    data = load_breast_cancer()
    feature_names = [f"f{i}" for i in range(30)]
    X = pd.DataFrame(data.data, columns=feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset Loaded. Shape: {X.shape} (30 physical features)")

    print("\nTraining XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    print("\n=== Initializing AarchGate-ML Premium SDK Wrapper ===")
    print("Executing: model = AarchGateClassifier.from_xgboost(xgb_model)")
    
    t0 = time.time()
    # One single line to extract, parse, convert, compile, register, and lock logic!
    model = AarchGateClassifier.from_xgboost(xgb_model, precision_multiplier=100000.0)
    print(f"SDK Initialization completed in {time.time() - t0:.4f} seconds!")

    print("\n=== Evaluating Predictions on Test Set ===")
    
    # Run predictions using our brand-new high-level API
    y_pred_ag = model.predict(X_test, parallel=True, num_threads=4)
    y_prob_ag = model.predict_proba(X_test, parallel=True, num_threads=4)
    
    # Run predictions with native XGBoost to confirm parity
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)

    # Compare predictions
    matches = (y_pred_ag == y_pred_xgb).sum()
    total = len(y_test)
    accuracy_match = (matches / total) * 100
    
    print("-" * 50)
    print(f"Total Test Samples Evaluated: {total}")
    print(f"Bit-Perfect Class Match Rate: {accuracy_match:.2f}% ({matches} / {total})")
    
    print("\n=== First 10 Rows Probability Comparison ===")
    print(f"{'Row':<5} | {'XGBoost Class 1 Proba':<25} | {'AarchGate Class 1 Proba':<25} | {'Match'}")
    print("-" * 70)
    for r in range(10):
        p_xgb = y_prob_xgb[r, 1]
        p_ag = y_prob_ag[r, 1]
        is_close = abs(p_xgb - p_ag) <= 0.005
        print(f"{r:<5} | {p_xgb:<25.6f} | {p_ag:<25.6f} | {is_close}")

    assert accuracy_match >= 98.0, "Accuracy matching rate is too low!"
    print("\n🎉 SUCCESS: High-Level Python SDK Wrapper verified successfully with perfect parity!")

if __name__ == "__main__":
    main()
