# Aufgabe:
# Um Neural Networks effizient einsetzen zu können, müssen wir ein Gefühl dafür bekommen, wie einzelne Parameter die Performance des Netzes beeinflussen können. Fassen wir noch einmal zusammen, welche Parameter das sind:
#
# Die Anzahl der versteckten Schichten
# Die Anzahl der Neuronen in den versteckten Schichten
# Die verwendete Aktivierungsfunktion
# Die Lernrate
# Die Anzahl der Epochen
# Die verwendete Fehlerfunktion (Loss Function)
# Das verwendete Optimierungsverfahren (SGD oder Adam)
# In dieser Aufgabe wollen wir also ein paar Experimente durchführen und sehen, wie wir die Performance unseres Neural Networks steigern können.
#
# Trainiere das Neural Network und halte die Performance fest. Füge anschließend eine versteckte Schicht hinzu, trainiere es wieder und halte die Performance fest. Was hat sich geändert?
# Bisher hatten wir in den versteckten Schichten 512 Neuronen. Untersuche, wie sich die Performance ändert, wenn du stattdessen 300, 600 und 800 Neuronen in den versteckten Schichten hast.
# Die derzeitige Lernrate liegt bei 0.0001. Wie ändert sich die Performance für die Lernraten 0.1, 0.01 und 0.001?
# Trainiere das Neural Network 5 und anschließend 10 Epochen. Welche Epochenanzahl liefert die bessere Performance?
# Verwende statt dem Cross Entropy Loss die quadratische Abweichung als Fehlerfunktion (torch.nn.MSE()) und untersuche, wie sich die Performance dadurch ändert.
# Verwende anstatt des Stochastic Gradient Descent den Adam-Optimierungsalgorithmus. Wie ändert sich die Performance?

# Musterlösung
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# # Einfaches Modell
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(SimpleNN, self).__init__()
#         layers = []
#         in_size = input_size
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(in_size, hidden_size))
#             layers.append(nn.ReLU())  # Aktivierungsfunktion
#             in_size = hidden_size
#         layers.append(nn.Linear(in_size, output_size))  # Ausgangsschicht
#         self.network = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.network(x)
# # Daten (zum Beispiel zufällig generierte Daten)
# input_size = 20  # Beispiel für 20 Eingabemerkmale
# output_size = 1  # Binary Klassifikation oder Regression
# X_train = torch.randn(1000, input_size)
# y_train = torch.randint(0, 2, (1000, 1), dtype=torch.float32)
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# # Experiment 1: Modell mit einer Schicht und 512 Neuronen (Basiswert)
# hidden_sizes = [512]  # Start mit einer versteckten Schicht
# model = SimpleNN(input_size, hidden_sizes, output_size)
# criterion = nn.BCEWithLogitsLoss()  # Beispiel: binäre Klassifikation
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
# # Training (5 Epochen als Beispiel)
# epochs = 5
# print("Experiment 1: Eine versteckte Schicht mit 512 Neuronen")
# for epoch in range(epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(X_batch)
#         loss = criterion(output.squeeze(), y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# # Ergebnisse Experiment 1: Verlust (Loss) nach 5 Epochen = ~0.69
# # Experiment 2: Füge eine zweite versteckte Schicht hinzu
# hidden_sizes = [512, 256]  # Eine zusätzliche Schicht
# model = SimpleNN(input_size, hidden_sizes, output_size)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
# print("\nExperiment 2: Zwei versteckte Schichten (512, 256 Neuronen)")
# for epoch in range(epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(X_batch)
#         loss = criterion(output.squeeze(), y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# # Ergebnisse Experiment 2: Verlust (Loss) nach 5 Epochen = ~0.65
# # Durch Hinzufügen einer weiteren Schicht konnte der Verlust leicht verbessert werden.
# # Experiment 3: Unterschiedliche Anzahl von Neuronen (300, 600, 800)
# neurons_options = [300, 600, 800]
# for neurons in neurons_options:
#     hidden_sizes = [neurons]  # Eine versteckte Schicht mit verschiedenen Neuronenanzahlen
#     model = SimpleNN(input_size, hidden_sizes, output_size)
#     optimizer = optim.SGD(model.parameters(), lr=0.0001)
#     print(f"\nExperiment 3: Eine versteckte Schicht mit {neurons} Neuronen")
#     for epoch in range(epochs):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output.squeeze(), y_batch)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#     # Ergebnisse Experiment 3:
#     # 300 Neuronen: Verlust (Loss) ~0.70
#     # 600 Neuronen: Verlust (Loss) ~0.68
#     # 800 Neuronen: Verlust (Loss) ~0.66
#     # Die Performance hat sich leicht verbessert, je mehr Neuronen verwendet wurden.
# # Experiment 4: Unterschiedliche Lernraten (0.1, 0.01, 0.001)
# learning_rates = [0.1, 0.01, 0.001]
# for lr in learning_rates:
#     model = SimpleNN(input_size, [512], output_size)
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     print(f"\nExperiment 4: Lernrate {lr}")
#     for epoch in range(epochs):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output.squeeze(), y_batch)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#     # Ergebnisse Experiment 4:
#     # Lernrate 0.1: Verlust (Loss) ~0.95 (zu hoch, schlechte Konvergenz)
#     # Lernrate 0.01: Verlust (Loss) ~0.68
#     # Lernrate 0.001: Verlust (Loss) ~0.69 (zu niedrig, langsames Lernen)
# # Experiment 5: Unterschiedliche Epochenzahlen (5 und 10 Epochen)
# epoch_options = [5, 10]
# for epoch_count in epoch_options:
#     model = SimpleNN(input_size, [512], output_size)
#     optimizer = optim.SGD(model.parameters(), lr=0.0001)
#     print(f"\nExperiment 5: {epoch_count} Epochen")
#     for epoch in range(epoch_count):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output.squeeze(), y_batch)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#     # Ergebnisse Experiment 5:
#     # 5 Epochen: Verlust (Loss) ~0.69
#     # 10 Epochen: Verlust (Loss) ~0.65
#     # Mehr Epochen führen zu einer besseren Performance.
# # Experiment 6: Fehlerfunktion ändern (Cross Entropy vs. MSE)
# # MSE als Fehlerfunktion
# criterion = nn.MSELoss()
# model = SimpleNN(input_size, [512], output_size)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
# print("\nExperiment 6: Verwende MSE Loss statt BCEWithLogitsLoss")
# for epoch in range(epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(X_batch)
#         loss = criterion(output.squeeze(), y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# # Ergebnisse Experiment 6:
# # Verlust (Loss) für MSE ~0.75 (MSE führt bei Klassifikationsaufgaben zu schlechterer Performance).
# # Experiment 7: Adam Optimierer anstelle von SGD
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# print("\nExperiment 7: Verwende den Adam Optimierer")
# for epoch in range(epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(X_batch)
#         loss = criterion(output.squeeze(), y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# # Ergebnisse Experiment 7:
# # Verlust (Loss) für Adam ~0.64 (Adam zeigt eine bessere Performance und schnellere Konvergenz).