import numpy as np

def slicer1(X, num_fields):
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

def slicer2(X, num_fields):
    num_blocks = X.shape[0] // 64
    out = np.zeros((num_blocks, num_fields, 64), dtype=np.uint64)
    shifts = np.arange(64, dtype=np.uint64)
    
    batch_blocks = 5000
    for start_b in range(0, num_blocks, batch_blocks):
        end_b = min(start_b + batch_blocks, num_blocks)
        count_b = end_b - start_b
        block_X = X[start_b * 64 : end_b * 64].reshape(count_b, 64, num_fields)
        
        for f in range(num_fields):
            cols = block_X[:, :, f]
            bits = (cols[:, :, None] >> shifts) & 1
            packed = np.sum(bits << shifts[None, :, None], axis=1)
            out[start_b:end_b, f, :] = packed
            
    return out.ravel()

# Test on 10000 rows
np.random.seed(42)
X = np.random.randint(0, 50000, (6400, 5), dtype=np.uint64)
out1 = slicer1(X, 5)
out2 = slicer2(X, 5)

print("Are outputs identical?", np.array_equal(out1, out2))
