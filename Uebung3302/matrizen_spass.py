import numpy as np
import scipy.special as sp

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# Gewichte zwischen input und hidden
w_ih = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])

# Gewichte zwischen hidden und output
w_ho = np.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])

input_vector = np.array([[0.9], [0.1], [0.8]])

# berechne den Inputvektor für das hidden Layer
input_hidden = np.dot(w_ih, input_vector)

# berechne den Outputvektor aus dem hidden layer
output_hidden = sp.expit(input_hidden)

#berechne den Inputvektor für die Ausgangsschicht
input_final = np.dot(w_ho, output_hidden)

# berechne den Outputvektor der Ausgangsschicht
output_final = sp.expit(input_final)

print(output_final)