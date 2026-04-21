import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("K4.0026_3.5.Ü.01_mnist_data.csv")
data_liste = data.values.tolist()
# Ersten vier Datensätze durchgehen
for i in range(4):
    eintrag = data_liste[i]
    label = eintrag[0]                   # Erste Zahl ist das Label
    pixel = np.array(eintrag[1:])        # Rest sind Pixelwerte
    bild = pixel.reshape(28, 28)         # In 28x28 umformen

    # Bild anzeigen
    plt.subplot(1, 4, i+1)
    plt.imshow(bild, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
