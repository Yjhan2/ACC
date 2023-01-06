import torch
import numpy as np

a = 'val'
b = 'id'
z = np.zeros((2))
d = torch.tensor([[1,2,3],[4,5,6]])
e = d.flip()
print(z)
print(torch.cuda.is_available())
print(a+b)