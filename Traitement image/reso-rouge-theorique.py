import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lmfit import Model

# Ouvrir le fichier et lire son contenu
with open("Traitement image/Red_laser_Small.txt", "r") as file:
    lines = file.readlines()

# Initialiser des listes pour les longueurs d'onde et les intensités
longueurs_donde = []
intensites = []

# Parcourir les lignes du fichier
for line in lines:
    # Ignorer les lignes de métadonnées
    if line.startswith(">>>>>Begin Spectral Data<<<<<") or line.startswith("Date:") or line.startswith("User:"):
        continue
    
    # Séparer chaque ligne en deux colonnes
    # Utiliser la séparation par tabulation ou espace pour diviser les valeurs
    parts = line.split()
    if len(parts) == 2:
        # Convertir les valeurs en float et les ajouter aux listes
        try:
            longueur_donde = float(parts[0])
            intensite = float(parts[1])
            longueurs_donde.append(longueur_donde)
            intensites.append(intensite)
        except ValueError:
            pass  # Si la conversion échoue, ignorer la ligne

print(f'intensité maximale rouge à : {longueurs_donde[np.argmax(intensites)]} nm')

# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(longueurs_donde,intensites/np.max(intensites))
plt.xlabel("Position horizontale (nm)")
plt.ylabel("Intensité normalisée")
plt.title("Intensité du laser rouge obtenue avec un spectromètre commercial")
plt.show()

def calculer_FWHM(x, y):
    # 1. Identifier la valeur maximale de l'intensité
    I_max = np.max(y)
    
    # 2. Calculer la moitié de cette valeur
    I_half = I_max / 2
    
    # 3. Trouver les indices où l'intensité atteint la moitié du maximum
    indices_fwhm = np.where(y >= I_half)[0]
    
    # 4. La FWHM est la distance entre les deux indices extrêmes
    if len(indices_fwhm) >= 2:
        x_fwhm_min = x[indices_fwhm[0]]
        x_fwhm_max = x[indices_fwhm[-1]]
        fwhm = x_fwhm_max - x_fwhm_min
    else:
        fwhm = None
        x_fwhm_min, x_fwhm_max = None, None
    
    return fwhm, x_fwhm_min, x_fwhm_max, I_half

# Exemple d'utilisation avec les données d'intensité rouge (introuge) et de longueur d'onde (val_lamda)
fwhm, x_fwhm_min, x_fwhm_max, I_half = calculer_FWHM(longueurs_donde, intensites)

print(f"FWHM: {fwhm} nm, à partir de {x_fwhm_min} nm à {x_fwhm_max} nm")

# Visualisation avec les lignes représentant la FWHM
plt.plot(longueurs_donde, intensites, label="Intensité rouge", color="red")
plt.axvline(x=x_fwhm_min, color='black', linestyle='--', label="Limite inférieure FWHM")
plt.axvline(x=x_fwhm_max, color='black', linestyle='--', label="Limite supérieure FWHM")
plt.axhline(y=I_half, color='blue', linestyle='--', label="Moitié de la hauteur maximale")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité")
plt.title("Calcul de la FWHM sur la courbe d'intensité")
plt.legend()
plt.grid(True)
plt.show()