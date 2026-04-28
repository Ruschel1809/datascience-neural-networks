import torch

shape =(3,3,)
rand_tensor = torch.rand(shape)
print(rand_tensor)
print(rand_tensor + 3)

new_tensor = torch.add(rand_tensor, 3)
print(new_tensor)
