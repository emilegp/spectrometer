import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
#image = Image.open("Photo-laser-blanche-13um-table.tif").convert("L")
image = Image.open("Photo-laser-rouge-atténué-n0=0.6-5um-table.tif").convert("L")
image_array = np.array(image)


# Calculer la moyenne d'intensité par colonne
intensité_par_colonne = np.mean(image_array, axis=0)

# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(intensité_par_colonne)
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité en fonction de la position horizontale")
plt.show()

