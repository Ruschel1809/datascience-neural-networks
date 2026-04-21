import torch

shape =(3,3,)
rand_tensor = torch.rand(shape)
print(rand_tensor)
print(rand_tensor + 3)

# Musterlösung
# Wir verwenden die torch.add()-Funktion. Diese hat die folgenden Parameter:
#
# Input: Der Tensor, dessen Elemente mit 3 addiert werden sollen
# Other: Der Tensor oder die Zahl, die zu dem Input-Tensor addiert werden soll
# Alpha: Der Vor-Faktor des Other-Parameters
# Der Output der Funktion liefert also folgendes Ergebnis:

# out(i) = input(i) + alpha x other(i)

# Wenden wir das nun auf die Aufgabenstellung an:

new_tensor = torch.add(rand_tensor, 3)
print(new_tensor)
