import aarchgate_python as _apex

class ApexEngine:
    def __init__(self):
        self._engine = _apex.ApexEngine()
        
    def register_schema(self, name, fields, stride):
        return _apex.register_schema(self._engine, name, fields, stride)
        
    def set_logic(self, schema_name, ir_root, mode):
        return _apex.set_logic(self._engine, schema_name, ir_root, mode)
        
    def execute(self, data, count, parallel=False, num_threads=4):
        if parallel:
            return _apex.execute_parallel(self._engine, data, count, num_threads)
        return _apex.execute(self._engine, data, count)

# Builder functions (proxies)
def builder_Load(name):
    return _apex.builder_Load(name)

def builder_Const(value):
    return _apex.builder_Const(value)

def builder_Add(a, b):
    return _apex.builder_Add(a, b)

def builder_GT(a, b):
    return _apex.builder_GT(a, b)

def builder_LT(a, b):
    return _apex.builder_LT(a, b)

def builder_GE(a, b):
    return _apex.builder_GE(a, b)

def builder_AND(a, b):
    return _apex.builder_AND(a, b)

def builder_Select(cond, a, b):
    return _apex.builder_Select(cond, a, b)

def builder_Sum(operands):
    return _apex.builder_Sum(operands)

def builder_Not(a):
    return _apex.builder_Not(a)

def builder_SetWeight(node, weight):
    return _apex.builder_SetWeight(node, weight)
