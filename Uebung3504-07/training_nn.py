#----------------------------------------------------------------
# Vorwärtsdurchlauf (Forward Propagation):
#
# Input → Hidden Layer → Output Layer (Signal berechnen)
#
# Fehler berechnen (Output vs. Label)
#
# Fehler rückwärts verteilen (Backpropagation) – d.h. Fehler in den Hidden Layer „übersetzen“
#
# Gewichte anpassen mit Lernrate
#----------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("K4.0026_3.5.Ü.01_mnist_data.csv")
data_liste = data.values.tolist()

# Datensatz durchmischen (wichtig für sinnvolle Aufteilung)
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

# Skalieren
# pixel/ 255 normiert den Pixelwert von 0-255 auf 0-1
# * 0.99 skaliert das Ergebnis auf 0-0.99
# + 0.01 verschiebt es in den Bereich 0.01-1.0
train_pixels = (train_pixels / 255.0) * 0.99 + 0.01
test_pixels = (test_pixels / 255.0) * 0.99 + 0.01

print("Trainingsdaten:", train_pixels.shape)
print("Testdaten:", test_pixels.shape)
print("Skalierter Pixelbereich:", train_pixels.min(), "bis", train_pixels.max())

# drei-schichtiges Neural Network
input_neuron = 784
hidden_neuron = 100
output_neuron = 10
lernrate = 0.3  # wird später im Training verwendet

# Gewichtsmatrizen initialisieren (Zufallswerte gleichverteilt zwischen -0.5 und 0.5) mit der Größe hidden_neuron x input_neuron bzw. output_neuron x hidden_neuron
w_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_neuron, input_neuron))
w_hidden_output = np.random.uniform(-0.5, 0.5, (output_neuron, hidden_neuron))

#Sigmoid-Aktivierungsfunktion für nichtlineare Transformation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Vorwärtsauswertung
def test(input_v, weights_ih, weights_ho):

    # Umformen Liste in Spaltenvektor
    inputs = np.array(input_v, ndmin=2).T  #2-D Array transponiert

    # Signale zum Hidden Layer
    hidden_inputs = np.dot(weights_ih, inputs) #Berechnung der gewichteten Summe der Eingabewerte für jedes Neuron im Hidden Layer
    hidden_outputs = sigmoid(hidden_inputs) #Anwendung der Aktivierungsfunktion (Sigmoid), um die Ausgabe jedes Neurons im Hidden Layer zu berechnen

    # Signale zum Output Layer
    final_inputs = np.dot(weights_ho, hidden_outputs) # Berechnung der gewichteten Ausgabe aus dem Hidden Layer für jedes Output-Neuron
    final_outputs = sigmoid(final_inputs) #Anwendung der Aktivierungsfunktion, um die finale Ausgabe des Netzwerks zu berechnen

    return final_outputs

def train(input_vector, w_input_hidden, w_hidden_output, target_vector, learning_rate):
    # Vorwärtsdurchlauf
    input_vector = np.array(input_vector, ndmin=2).T  # Spaltenvektor (784,1)
    target_vector = np.array(target_vector, ndmin=2).T  # Spaltenvektor (10,1)

    # Input → Hidden
    hidden_inputs = np.dot(w_input_hidden, input_vector)  # (100,784) · (784,1) = (100,1)
    hidden_outputs = sigmoid(hidden_inputs)  # (100,1)

    # Hidden → Output
    final_inputs = np.dot(w_hidden_output, hidden_outputs)  # (10,100) · (100,1) = (10,1)
    final_outputs = sigmoid(final_inputs)  # (10,1)

    # Fehler (Ziel - Ausgabe)
    output_errors = target_vector - final_outputs  # (10,1)

    # Fehler ins Hidden Layer zurückpropagieren
    hidden_errors = np.dot(w_hidden_output.T, output_errors)  # (100,10) · (10,1) = (100,1)

    # Gewichte aktualisieren (Output → Hidden)
    w_hidden_output += learning_rate * np.dot(
        (output_errors * sigmoid_derivative(final_outputs)),
        hidden_outputs.T
    )

    # Gewichte aktualisieren (Hidden → Input)
    w_input_hidden += learning_rate * np.dot(
        (hidden_errors * sigmoid_derivative(hidden_outputs)),
        input_vector.T
    )

    return w_input_hidden, w_hidden_output

# Erstes Element als Eingabe
input_vektor = train_pixels[0]

# Netzwerk testen
output_vektor = test(input_vektor, w_input_hidden, w_hidden_output)

anzahl_epochen = 5  # z.B. 5 Durchläufe durch die Trainingsdaten

for epoch in range(anzahl_epochen):
    print(f"Epoche {epoch + 1}/{anzahl_epochen}")
    for i in range(len(train_pixels)):
        input_vector = train_pixels[i]
        label = train_labels[i]

        # Zielvektor erstellen
        target_vector = np.zeros((10)) + 0.01
        target_vector[label] = 0.99

        # Netz trainieren
        w_input_hidden, w_hidden_output = train(
            input_vector,
            w_input_hidden,
            w_hidden_output,
            target_vector,
            lernrate
        )


korrekt = 0
gesamt = len(test_pixels)

# Netzwerk testen
for i in range(gesamt):
    input_vector = test_pixels[i]
    label = test_labels[i]

    output_vector = test(input_vector, w_input_hidden, w_hidden_output)
    vorhergesagt = np.argmax(output_vector)

    if vorhergesagt == label:
        korrekt += 1

genauigkeit = korrekt / gesamt
print(f"Genauigkeit: {genauigkeit * 100:.2f}%")

# Ausgabe
print("Output-Vektor:", output_vektor)
print("Vorhergesagte Ziffer:", np.argmax(output_vektor))
print("Anzahl korrekte Vorhersagen: ", korrekt)

performance = korrekt / gesamt
print(f"Performance (Accuracy): {performance:.4f} ({korrekt}/{gesamt} richtig)")

# Bild anzeigen für den vorhergesagten Output
plt.imshow(input_vektor.reshape(28, 28), cmap='Greys')
plt.title(f"Vorhergesagte Ziffer: {np.argmax(output_vektor)}")
plt.axis('off')
plt.show()

# TODO Änderungen für Aufgabe 3507 einbauen
# Wir verpacken das Training und Testen in eine Funktion train_and_evaluate.

# def train_and_evaluate(train_pixels, train_labels, test_pixels, test_labels, learning_rate, epochs=5):
#     # Netzwerkparameter
#     input_neuron = 784
#     hidden_neuron = 100
#     output_neuron = 10
#
#     # Initialisiere Gewichte für jede Lernrate neu
#     w_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_neuron, input_neuron))
#     w_hidden_output = np.random.uniform(-0.5, 0.5, (output_neuron, hidden_neuron))
#
#     # Training
#     for epoch in range(epochs):
#         for i in range(len(train_pixels)):
#             input_vector = train_pixels[i]
#             label = train_labels[i]
#
#             # Zielvektor erstellen
#             target_vector = np.zeros((10)) + 0.01
#             target_vector[label] = 0.99
#
#             w_input_hidden, w_hidden_output = train(
#                 input_vector,
#                 w_input_hidden,
#                 w_hidden_output,
#                 target_vector,
#                 learning_rate
#             )
#
#     # Testen
#     korrekt = 0
#     for i in range(len(test_pixels)):
#         input_vector = test_pixels[i]
#         label = test_labels[i]
#
#         output_vector = test(input_vector, w_input_hidden, w_hidden_output)
#         vorhergesagt = np.argmax(output_vector)
#
#         if vorhergesagt == label:
#             korrekt += 1
#
#     genauigkeit = korrekt / len(test_pixels)
#     return genauigkeit
#
# 🧪 2. Verschiedene Lernraten testen
#
# lernrate_liste = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# performances = []
#
# for lr in lernrate_liste:
#     print(f"Trainiere mit Lernrate {lr}")
#     accuracy = train_and_evaluate(train_pixels, train_labels, test_pixels, test_labels, learning_rate=lr, epochs=5)
#     performances.append(accuracy)
#     print(f"Genauigkeit: {accuracy * 100:.2f}%\n")
# 📊 3. Ergebnisse plotten
# plt.figure(figsize=(10, 6))
# plt.plot(lernrate_liste, performances, marker='o')
# plt.xlabel("Lernrate")
# plt.ylabel("Genauigkeit (Accuracy)")
# plt.title("Genauigkeit in Abhängigkeit von der Lernrate")
# plt.grid(True)
# plt.show()
#4. Optimale Lernrate bestimmen
# beste_genauigkeit = max(performances)
# beste_lr_index = performances.index(beste_genauigkeit)
# beste_lr = lernrate_liste[beste_lr_index]
#
# print(f"Beste Lernrate: {beste_lr} mit einer Genauigkeit von {beste_genauigkeit * 100:.2f}%")