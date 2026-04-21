import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

start = time.time()
end = time.time()
print("Runtime", end-start, " s")

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

input_size = 784 # Anzahl der Pixel jedes Bildes
output_size = 10 # die Daten lassen sich in 10 Klassen einteilen
learning_rate = 0.0001
num_epochs = 5

# Daten importieren
xy = np.loadtxt("K4.0026_3.5.Ü.01_mnist_data.csv", delimiter=",", dtype=np.float32, skiprows=1)

# 2. Features und Labels aufteilen
X_features = xy[:, 1:]  # Alle Spalten ab der 2. sind die Features (Pixel)
y_labels = xy[:, [0]] # Erste Spalte ist das Label

# 3. In torch Tensors umwandeln
X_tensor = torch.from_numpy(X_features)
y_tensor = torch.from_numpy(y_labels).long().squeeze()  # squeeze() um 2D -> 1D zu machen

# 4. Dataset erstellen
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

# 5. Dataset zufällig in 80% / 20% splitten
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 6. Dataloader erstellen
train_loader = DataLoader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True)

model = NN(input_size, output_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):

        labels = labels.type(torch.LongTensor)
        # Daten durchs Netz jagen
        outputs = model(data)
        loss = loss_fn(outputs, labels)

        # Backpropagierung
        optimizer.zero_grad() # bei Beginn jedes Schleifendurchlaufs Gradienten auf 0
        loss.backward()
        optimizer.step() # Gradientenverfahren

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f"loss: {loss} Durchläufe: {current}")

richtige = []
for idx, (data, label) in enumerate(test_loader):

    output = model(data)
    output = output.argmax()

    if label == output:
        richtige.append(1)
    else:
        richtige.append(0)
performance = sum(richtige)/test_size
print(test_size)
print(sum(richtige))
print(performance)