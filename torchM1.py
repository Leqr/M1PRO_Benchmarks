import torch
import time
import numpy as np

shape = [5000,5000]
A_torch = torch.rand(shape, device="mps")
n = 10000
start_torch = time.perf_counter()
for i in range(n):
    A_torch = A_torch*A_torch
t_torch = time.perf_counter() - start_torch
print(f"Torch time : {t_torch}")

torch.manual_seed(42)
shape = (300,300)
c = torch.rand(shape,device="mps")

runtimes = 10
timecosts = []
for _ in range(runtimes):
    s_time = time.perf_counter()
    for i in range(100):
        c += 1
        torch.svd(c)
    timecosts.append(time.perf_counter() - s_time)

print(f'mean of {runtimes} runs torch: {np.mean(timecosts):.5f}s')
t_torch = np.mean(timecosts)