import numpy as np
import json
import xgboost as xgb
import bindings.python.aarchgate as apex
from bindings.python.converter import XGBConverter

class AarchGateBaseEstimator:
    def __init__(self, precision_multiplier=100000.0, shift_negative=1000.0):
        self.precision_multiplier = precision_multiplier
        self.shift_negative = shift_negative
        self.engine = apex.ApexEngine()
        self.schema_name = f"schema_{id(self)}"
        self.feature_names = []

    def _prepare_data(self, X):
        # Convert pandas DataFrame or Series to numpy array if applicable
        if hasattr(X, "columns") and hasattr(X, "values"):
            if self.feature_names:
                missing = [f for f in self.feature_names if f not in X.columns]
                if missing:
                    X_np = X.values
                else:
                    X_np = X[self.feature_names].values
            else:
                X_np = X.values
        else:
            X_np = np.atleast_2d(X)

        num_rows, num_features = X_np.shape
        
        # Scaling and shifting
        # Shift negative numbers into positive space, then scale and cast to uint64_t
        X_scaled = np.round((X_np + self.shift_negative) * self.precision_multiplier).astype(np.uint64)
        
        # Return flattened uint8 view for raw byte extraction
        return np.ascontiguousarray(X_scaled).view(np.uint8).ravel(), num_rows

class AarchGateClassifier(AarchGateBaseEstimator):
    @classmethod
    def from_xgboost(cls, model_or_booster, precision_multiplier=100000.0, shift_negative=1000.0):
        inst = cls(precision_multiplier, shift_negative)
        
        if hasattr(model_or_booster, "get_booster"):
            booster = model_or_booster.get_booster()
        else:
            booster = model_or_booster
            
        model_json_str = booster.save_raw(raw_format="json").decode("utf-8")
        model_json = json.loads(model_json_str)
        
        converter = XGBConverter(precision_multiplier=precision_multiplier)
        converter._walk_tree = lambda tree, node_idx, current_cond=None: inst._custom_walk_tree(converter, tree, node_idx, current_cond)
        
        ir_root = converter.convert(model_json)
        inst.feature_names = converter.features
        num_features = len(inst.feature_names)
        
        fields = [(name, i * 8, 64, 0) for i, name in enumerate(inst.feature_names)]
        stride_bytes = num_features * 8
        inst.engine.register_schema(inst.schema_name, fields, stride_bytes)
        inst.engine.set_logic(inst.schema_name, ir_root, 0) # 0 = BIT_SLICED
        
        return inst

    def _custom_walk_tree(self, conv, tree, node_idx, current_cond=None):
        left_child = tree['left_children'][node_idx]
        right_child = tree['right_children'][node_idx]
        
        if left_child == -1:
            raw_weight = np.float32(tree['split_conditions'][node_idx])
            return [(current_cond, conv._quantize(raw_weight))]
            
        feature_idx = tree['split_indices'][node_idx]
        feature_name = conv.features[feature_idx]
        threshold = np.float32(tree['split_conditions'][node_idx])
        
        q_threshold = conv._quantize(threshold + np.float32(self.shift_negative))
        
        load_f = apex.builder_Load(feature_name)
        const_t = apex.builder_Const(q_threshold)
        is_lt = apex.builder_LT(load_f, const_t)
        is_ge = apex.builder_Not(is_lt)
        
        left_cond = is_lt if current_cond is None else apex.builder_AND(current_cond, is_lt)
        right_cond = is_ge if current_cond is None else apex.builder_AND(current_cond, is_ge)
        
        results = []
        results.extend(self._custom_walk_tree(conv, tree, left_child, left_cond))
        results.extend(self._custom_walk_tree(conv, tree, right_child, right_cond))
        return results

    def predict_proba(self, X, parallel=False, num_threads=4):
        data_bytes, num_rows = self._prepare_data(X)
        stride_bytes = len(self.feature_names) * 8
        
        margins = np.zeros(num_rows, dtype=np.float64)
        for r in range(num_rows):
            row_view = data_bytes[r * stride_bytes : (r + 1) * stride_bytes]
            raw_val = self.engine.execute(row_view, 1)
            val = raw_val if raw_val < 2**63 else raw_val - 2**64
            margins[r] = val / self.precision_multiplier
            
        p = 1.0 / (1.0 + np.exp(-margins))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, parallel=False, num_threads=4):
        prob = self.predict_proba(X, parallel, num_threads)
        return np.argmax(prob, axis=1)

class AarchGateRegressor(AarchGateBaseEstimator):
    @classmethod
    def from_xgboost(cls, model_or_booster, precision_multiplier=100000.0, shift_negative=1000.0):
        inst = cls(precision_multiplier, shift_negative)
        
        if hasattr(model_or_booster, "get_booster"):
            booster = model_or_booster.get_booster()
        else:
            booster = model_or_booster
            
        model_json_str = booster.save_raw(raw_format="json").decode("utf-8")
        model_json = json.loads(model_json_str)
        
        converter = XGBConverter(precision_multiplier=precision_multiplier)
        converter._walk_tree = lambda tree, node_idx, current_cond=None: inst._custom_walk_tree(converter, tree, node_idx, current_cond)
        
        ir_root = converter.convert(model_json)
        inst.feature_names = converter.features
        num_features = len(inst.feature_names)
        
        fields = [(name, i * 8, 64, 0) for i, name in enumerate(inst.feature_names)]
        stride_bytes = num_features * 8
        inst.engine.register_schema(inst.schema_name, fields, stride_bytes)
        inst.engine.set_logic(inst.schema_name, ir_root, 0) # BIT_SLICED
        
        return inst

    def _custom_walk_tree(self, conv, tree, node_idx, current_cond=None):
        left_child = tree['left_children'][node_idx]
        right_child = tree['right_children'][node_idx]
        
        if left_child == -1:
            raw_weight = np.float32(tree['split_conditions'][node_idx])
            return [(current_cond, conv._quantize(raw_weight))]
            
        feature_idx = tree['split_indices'][node_idx]
        feature_name = conv.features[feature_idx]
        threshold = np.float32(tree['split_conditions'][node_idx])
        
        q_threshold = conv._quantize(threshold + np.float32(self.shift_negative))
        
        load_f = apex.builder_Load(feature_name)
        const_t = apex.builder_Const(q_threshold)
        is_lt = apex.builder_LT(load_f, const_t)
        is_ge = apex.builder_Not(is_lt)
        
        left_cond = is_lt if current_cond is None else apex.builder_AND(current_cond, is_lt)
        right_cond = is_ge if current_cond is None else apex.builder_AND(current_cond, is_ge)
        
        results = []
        results.extend(self._custom_walk_tree(conv, tree, left_child, left_cond))
        results.extend(self._custom_walk_tree(conv, tree, right_child, right_cond))
        return results

    def predict(self, X, parallel=False, num_threads=4):
        data_bytes, num_rows = self._prepare_data(X)
        stride_bytes = len(self.feature_names) * 8
        
        preds = np.zeros(num_rows, dtype=np.float64)
        for r in range(num_rows):
            row_view = data_bytes[r * stride_bytes : (r + 1) * stride_bytes]
            raw_val = self.engine.execute(row_view, 1)
            val = raw_val if raw_val < 2**63 else raw_val - 2**64
            preds[r] = val / self.precision_multiplier
            
        return preds
