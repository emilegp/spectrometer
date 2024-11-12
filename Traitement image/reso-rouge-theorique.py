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







# Définir les modèles de fonctions gaussienne, lorentzienne, et sinc
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma):
    return (amplitude * gamma**2) / ((x - center)**2 + gamma**2)

def sinc(x, amplitude, center, width):
    # Calcul de sinc en normalisant (x - center) pour définir la largeur
    return amplitude * np.sinc((x - center) / width)

# Ajuster les données
x = np.arange(len(intensites))  # Positions des pixels

# Modèle gaussien
gauss_model = Model(gaussian)
gauss_result = gauss_model.fit(intensites, x=x, amplitude=np.max(intensites), center=np.argmax(intensites), sigma=1)
fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * gauss_result.params['sigma'].value

# Modèle lorentzien
lorentz_model = Model(lorentzian)
lorentz_result = lorentz_model.fit(intensites, x=x, amplitude=np.max(intensites), center=np.argmax(intensites), gamma=1)
fwhm_lorentz = 2 * lorentz_result.params['gamma'].value

# Modèle sinc
sinc_model = Model(sinc)
sinc_result = sinc_model.fit(intensites, x=x, amplitude=np.max(intensites), center=np.argmax(intensites), width=5)
# La FWHM d'une fonction sinc n'est pas aussi facilement définie que pour les gaussiennes et lorentziennes
# mais on pourrait estimer la largeur à mi-hauteur en fonction du paramètre `width`
fwhm_sinc = 2 * sinc_result.params['width'].value  # Approximation pour FWHM de sinc

# Imprimer les résolutions
print(f"Résolution gaussienne (FWHM) : {fwhm_gauss:.2f} pixels")
print(f"Résolution lorentzienne (FWHM) : {fwhm_lorentz:.2f} pixels")
print(f"Résolution sinc approximée (FWHM) : {fwhm_sinc:.2f} pixels")

# Afficher les statistiques pour chaque ajustement

# Statistiques pour l'ajustement gaussien
print("\nStatistiques de l'ajustement gaussien :")
print(f" - Chi-carré : {gauss_result.chisqr}")
print(f" - Chi-carré réduit : {gauss_result.redchi}")
print(f" - R² : {1 - (np.sum((intensites - gauss_result.best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)):.4f}")

# Statistiques pour l'ajustement lorentzien
print("\nStatistiques de l'ajustement lorentzien :")
print(f" - Chi-carré : {lorentz_result.chisqr}")
print(f" - Chi-carré réduit : {lorentz_result.redchi}")
print(f" - R² : {1 - (np.sum((intensites - lorentz_result.best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)):.4f}")

# Statistiques pour l'ajustement sinc
print("\nStatistiques de l'ajustement sinc :")
print(f" - Chi-carré : {sinc_result.chisqr}")
print(f" - Chi-carré réduit : {sinc_result.redchi}")
print(f" - R² : {1 - (np.sum((intensites - sinc_result.best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)):.4f}")

# Comparer les résultats et déterminer le meilleur ajustement
best_model = min(
    [
        ("gaussien", gauss_result.redchi, gauss_result),
        ("lorentzien", lorentz_result.redchi, lorentz_result),
        ("sinc", sinc_result.redchi, sinc_result),
    ],
    key=lambda item: (abs(item[1] - 1), -1 * (1 - np.sum((intensites - item[2].best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)))
)

print("\nComparaison des ajustements :")
print(f"Le meilleur choix pour ajuster les données est le modèle {best_model[0]}.")

# Visualisation des ajustements
plt.plot(x, intensites, label="Intensité bleue", color="blue")
plt.plot(x, gauss_result.best_fit, 'k--', label=f"Ajustement Gaussien (FWHM = {fwhm_gauss:.2f})")
plt.plot(x, lorentz_result.best_fit, 'g--', label=f"Ajustement Lorentzien (FWHM = {fwhm_lorentz:.2f})")
plt.plot(x, sinc_result.best_fit, 'r--', label=f"Ajustement Sinc (FWHM approx = {fwhm_sinc:.2f})")

# Ajouter des détails de légende et de mise en forme
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe d'intensité et ajustements gaussien, lorentzien et sinc")
plt.legend()
plt.show()