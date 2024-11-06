import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = Image.open("Photo-laser-rouge-atténué-n0=0.6-5um-table.tif").convert("L")
image_array = np.array(image)

TroisD=False

if TroisD:
    image_array=np.flip(image_array)

# Étape 1 : Trouver l'intensité maximale dans l'image
intensité_max = np.max(image_array)

# Étape 2 : Identifier les pixels ayant une intensité >= 90% de l'intensité maximale
seuil = 0.95 * intensité_max
mask = image_array >= seuil

# Étape 3 : Trouver les lignes contenant au moins un point au-dessus du seuil
lignes_à_considerer = np.any(mask, axis=1)

# Étape 4 : Sélectionner ces lignes et calculer la moyenne d'intensité par colonne
intensité_par_colonne = np.mean(image_array[lignes_à_considerer, :], axis=0)

# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(intensité_par_colonne)
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité (lignes sélectionnées)")
plt.show()
