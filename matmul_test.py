import tensorflow as tf
import torch 
import numpy as np
import time
import tabulate

shape = [5000,5000]
#float32
A_torch = torch.rand(shape)
A_numpy = A_torch.numpy()
A_tensor = tf.Variable(tf.convert_to_tensor(A_numpy))

print(A_torch.shape)
print(A_tensor.shape)
print(A_numpy.shape)
print(A_torch.type())
print(A_tensor.dtype)
print(type(A_numpy[0,0]))

n = 100
start_torch = time.perf_counter()
for i in range(n):
    A_torch = A_torch*A_torch
t_torch = time.perf_counter() - start_torch
print(f"Torch time : {t_torch}")

start_tf = time.perf_counter()
for i in range(n):
    A_tensor = A_tensor@A_tensor
t_tf = time.perf_counter() - start_tf
print(f"Tensorflow time : {t_tf}")

start_np = time.perf_counter()
for i in range(n):
    A_numpy = A_numpy@A_numpy
t_np = time.perf_counter() - start_np
print(f"Numpy time : {t_np}")

data = [["PyTorch",str(round(t_torch,6))+"s"],["Tensorflow",str(round(t_tf,6))+"s"],["Numpy",str(round(t_np,6))+"s"]]
print(tabulate.tabulate(data))

