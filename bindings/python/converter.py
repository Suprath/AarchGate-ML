import json
import numpy as np

try:
    import bindings.python.aarchgate as apex
except ImportError:
    # This will be resolved when running inside Docker with PYTHONPATH set
    apex = None

class XGBConverter:
    def __init__(self, precision_multiplier=1.0):
        self.precision_multiplier = precision_multiplier
        self.features = []

    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            return json.load(f)

    def _quantize(self, value):
        return int(value * self.precision_multiplier)

    def _walk_tree(self, tree, node_idx=0):
        left_child = tree['left_children'][node_idx]
        right_child = tree['right_children'][node_idx]
        
        if left_child == -1: # Leaf
            # Cast leaf weight to float32 to match XGBoost native precision before scaling
            raw_weight = np.float32(tree['split_conditions'][node_idx])
            return apex.builder_Const(self._quantize(raw_weight))
        
        feature_idx = tree['split_indices'][node_idx]
        # XGBoost internally uses float32 for thresholds. Cast to float32 before shifting.
        raw_cond = np.float32(tree['split_conditions'][node_idx])
        condition = self._quantize(raw_cond + np.float32(1000.0))
        
        left_node = self._walk_tree(tree, left_child)
        right_node = self._walk_tree(tree, right_child)
        
        feat_load = apex.builder_Load(f"f{feature_idx}")
        cond_const = apex.builder_Const(condition)
        
        # XGBoost: if feat < condition then left else right
        is_lt = apex.builder_LT(feat_load, cond_const)
        print(f"[CONVERTER DEBUG] Node {node_idx}: F{feature_idx} < {raw_cond} (quant: {condition}) -> Left:{left_child}, Right:{right_child}")
        return apex.builder_Select(is_lt, left_node, right_node)

    def convert(self, model_json):
        # model_json['learner']['gradient_booster']['model']['trees']
        try:
            trees = model_json['learner']['gradient_booster']['model']['trees']
        except KeyError:
            trees = model_json['learner']['gradient_booster']['model_tree_log']['trees']
        
        print(f"Converter: Found {len(trees)} trees in model.")
        
        # Extract global base score (margin)
        try:
            base_score_str = model_json['learner']['learner_model_param']['base_score']
            base_score = float(base_score_str.strip('[]'))
        except KeyError:
            base_score = 0.5
            
        tree_roots = []
        for tree in trees:
            tree_roots.append(self._walk_tree(tree))
        
        # Summation layer: recursive Add tree + base_score
        tree_sum = self._sum_trees(tree_roots)
        base_const = apex.builder_Const(self._quantize(base_score))
        return apex.builder_Add(tree_sum, base_const)

    def _sum_trees(self, roots):
        if not roots:
            return apex.builder_Const(0)
        if len(roots) == 1:
            return roots[0]
        
        mid = len(roots) // 2
        left_sum = self._sum_trees(roots[:mid])
        right_sum = self._sum_trees(roots[mid:])
        
        return apex.builder_Add(left_sum, right_sum)

def convert_xgboost(model_path, output_schema_name="xgboost_model"):
    converter = XGBConverter()
    model = converter.load_model(model_path)
    ir_root = converter.convert(model)
    return ir_root
