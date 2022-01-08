import time
import numpy as np
import tabulate 

np.random.seed(42)
a = np.random.rand(100000)
runtimes = 10
a.astype(np.csingle)

timecosts = []
for _ in range(runtimes):
    s_time = time.perf_counter()
    for i in range(100):
        a += 1
        np.fft.fft(a)
    timecosts.append(time.perf_counter() - s_time)

print(f'mean of {runtimes} runs numpy: {np.mean(timecosts):.5f}s')
t_np = np.mean(timecosts)
import tensorflow as tf
tf.random.set_seed(42)
b = tf.cast(a, tf.complex64)

timecosts = []
for _ in range(runtimes):
    s_time = time.perf_counter()
    for i in range(100):
        b += 1
        tf.signal.fft(b)
    timecosts.append(time.perf_counter() - s_time)

print(f'mean of {runtimes} runs tensorflow: {np.mean(timecosts):.5f}s')
t_tf = np.mean(timecosts)

import torch
torch.manual_seed(42)
c = torch.tensor(a,dtype=torch.complex64)

timecosts = []
for _ in range(runtimes):
    s_time = time.perf_counter()
    for i in range(100):
        c += 1
        torch.fft.fft(c)
    timecosts.append(time.perf_counter() - s_time)

print(f'mean of {runtimes} runs torch: {np.mean(timecosts):.5f}s')
t_torch = np.mean(timecosts)

data = [["PyTorch",str(round(t_torch,6))+"s"],["Tensorflow",str(round(t_tf,6))+"s"],["Numpy",str(round(t_np,6))+"s"]]
print(tabulate.tabulate(data))

