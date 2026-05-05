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
        return int(round(value * self.precision_multiplier))

    def _walk_tree(self, tree, node_idx, current_cond=None):
        left_child = tree['left_children'][node_idx]
        right_child = tree['right_children'][node_idx]
        
        if left_child == -1: # Leaf
            raw_weight = np.float32(tree['split_conditions'][node_idx])
            # For the base leaf, we can return (None, weight)
            return [(current_cond, self._quantize(raw_weight))]
        
        feature_idx = tree['split_indices'][node_idx]
        feature_name = self.features[feature_idx]
        threshold = np.float32(tree['split_conditions'][node_idx])
        
        # Cast threshold to float32 to match XGBoost native precision before scaling
        # Shift by +1000.0 to match the data transformation
        q_threshold = self._quantize(threshold + np.float32(1000.0))
        
        load_f = apex.builder_Load(feature_name)
        const_t = apex.builder_Const(q_threshold)
        
        is_lt = apex.builder_LT(load_f, const_t)
        is_ge = apex.builder_Not(is_lt)
        
        left_cond = is_lt if current_cond is None else apex.builder_AND(current_cond, is_lt)
        right_cond = is_ge if current_cond is None else apex.builder_AND(current_cond, is_ge)
        
        results = []
        results.extend(self._walk_tree(tree, left_child, left_cond))
        results.extend(self._walk_tree(tree, right_child, right_cond))
        return results

    def convert(self, model_json):
        learner = model_json['learner']
        gradient_booster = learner['gradient_booster']
        model_type = gradient_booster.get('model_type', 'gbtree')
        
        if model_type != 'gbtree':
            raise ValueError(f"Unsupported model type: {model_type}")
            
        trees = gradient_booster['model']['trees']
        base_score_str = learner['learner_model_param']['base_score']
        base_score = float(base_score_str.strip('[]'))
        
        # Extract feature names
        try:
            self.features = learner['feature_names']
        except KeyError:
            # Fallback if names are missing
            num_features = int(learner['learner_model_param']['num_feature'])
            self.features = [f"f{i}" for i in range(num_features)]
            
        print(f"Converter: Found {len(trees)} trees in model.")
        
        all_indicators = []
        total_base_weight = self._quantize(base_score)
        
        for tree in trees:
            leaves = self._walk_tree(tree, 0)
            for cond, weight in leaves:
                if cond is None:
                    total_base_weight += weight
                else:
                    # SUM(popcount(cond) * weight)
                    indicator = cond
                    apex.builder_SetWeight(indicator, weight)
                    all_indicators.append(indicator)
        
        return apex.builder_Add(apex.builder_Sum(all_indicators), apex.builder_Const(total_base_weight))

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
