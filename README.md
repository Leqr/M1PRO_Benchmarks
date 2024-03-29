# M1PRO_Benchmarks
On Apple M1 Pro 8-core with 16 GB of RAM and 14-core GPU. 

## Python
### Numpy vs TensorFlow vs PyTorch
Numpy (not using Accelerate/vecLib, see https://developer.apple.com/forums/thread/695963) and PyTorch installed through conda-forge with ARM64 support, TensorFlow-metal with apple silicon gpu support.

#### Matrix Mulitplication
```
matmul_test.py
```
Numpy throws ```RuntimeWarning: overflow encountered in matmul A_numpy = A_numpy@A_numpy```

32-bit floats

PyTorch  | Tensorflow | Numpy conda-forge   | PyTorch Apple Silicon  |
-------- | -----------|---------------------|------------------------|
0.497199s| 0.04535s  | 60.797036s           | 0.008s

#### Singular Value Decomposition
```
svd_test.py
```
PyTorch  | Tensorflow | Numpy conda-forge | Numpy vecLib compiled
-------- | -----------|------------------ | ---------------------
3.440609s| 1.095954s  | 4.545333s         | 1.04695s


