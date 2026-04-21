import numpy as np
import pandas as pd

data = pd.read_csv("K4.0026_3.5.Ü.01_mnist_data.csv")
data_liste = data.values.tolist()

# Datensatz durchmischen (sehr wichtig für sinnvolle Aufteilung!)
np.random.shuffle(data_liste)

# Aufteilen in 80 % des Datensatzes zum Training und 20 % zum Test
anzahl_gesamt = len(data_liste)
anzahl_train = int(anzahl_gesamt * 0.8)

train_data = data_liste[:anzahl_train]
test_data = data_liste[anzahl_train:]

# Trainingsdaten
train_labels = [eintrag[0] for eintrag in train_data]
train_pixels = [eintrag[1:] for eintrag in train_data]

# Testdaten
test_labels = [eintrag[0] for eintrag in test_data]
test_pixels = [eintrag[1:] for eintrag in test_data]

# In NumPy-Arrays umwandeln für bessere Performance
train_pixels = np.asarray(train_pixels, dtype=float)
test_pixels = np.asarray(test_pixels, dtype=float)
test_pixels = np.asarray(test_pixels)

# Skalieren
# pixel/ 255 normiert den Pixelwert von 0-255 auf 0-1
# * 0.99 skaliert das Ergebnis auf 0-0.99
# + 0.01 verschiebt es in den Bereich 0.01-1.0
train_pixels = (train_pixels / 255.0) * 0.99 + 0.01
test_pixels = (test_pixels / 255.0) * 0.99 + 0.01

print("Trainingsdaten:", train_pixels.shape)
print("Testdaten:", test_pixels.shape)
print("Skalierter Pixelbereich:", train_pixels.min(), "bis", train_pixels.max())