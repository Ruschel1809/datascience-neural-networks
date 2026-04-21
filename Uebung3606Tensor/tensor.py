import torch

shape=(4,4,)
tensor=torch.rand(shape)
print(tensor)

#letzte Spalte
print(tensor[:,-1])

#Element in 3. Zeile und 3. Spalte
print(tensor[2,2])

tensor[:,1]=0
print(tensor)