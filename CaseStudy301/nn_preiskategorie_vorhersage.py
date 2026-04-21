import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Schritt 1: Daten einlesen

df = pd.read_csv('K4.0026_2.C.02_MobilePhone.csv')

# Schritt 2: Features und Label trennen
X = df.drop('Price Range', axis=1)
y = df['Price Range']

# Schritt 3: Labels in Zahlen umwandeln (l=0, m=1, h=2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Schritt 4: Feature-Normalisierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Schritt 5: Tensoren vorbereiten
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Schritt 6: Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# NN-Klasse
class PhonePriceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PhonePriceClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Netzwerk initialisieren
input_size = X.shape[1]
hidden_size = 16  # kannst du später variieren
num_classes = len(label_encoder.classes_)
model = PhonePriceClassifier(input_size, hidden_size, num_classes)

# Trainingsparameter
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Training
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    acc = accuracy_score(y_test, predicted)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

# KNN Vergleich
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train.numpy(), y_train.numpy())
knn_preds = knn.predict(X_test.numpy())

knn_acc = accuracy_score(y_test.numpy(), knn_preds)
print(f"KNN Accuracy: {knn_acc * 100:.2f}%")

# Schritt 5: Modell-Tuning (optional)
#
# Du kannst z.B. ändern:
#
# hidden_size: mehr/weniger Neuronen
#
# epochs: länger trainieren
#
# lr: Lernrate ändern
#
# optimizer: z.B. SGD statt Adam
#
# activation: ReLU → LeakyReLU, Tanh, etc.
#
# loss function: ggf. mit gewichteter CrossEntropyLoss bei unbalancierten Klassen
#
# Beispiel mit zusätzlicher Schicht und anderer Aktivierungsfuntion:

# self.model = nn.Sequential(
#     nn.Linear(input_size, 32),
#     nn.LeakyReLU(),
#     nn.Linear(32, hidden_size),
#     nn.ReLU(),
#     nn.Linear(hidden_size, num_classes)
# )