import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lmfit import Model

# Charger les images en niveaux de gris
image = Image.open("Spectro-laser_bleu-lames-3D.tif").convert("L")
image_array = np.array(image)

def intensite(matrice_image):
    intensité_max = np.max(matrice_image)
    if intensité_max == 255:
        print("L'image est saturée")
        return

    seuil = 0.95 * intensité_max
    mask = matrice_image >= seuil
    lignes_à_considerer = np.any(mask, axis=1)
    intensité_par_colonne = np.mean(matrice_image[lignes_à_considerer, :], axis=0)

    return intensité_par_colonne

# Extraire les courbes d'intensité
int = intensite(image_array)

# Définir les modèles de fonctions gaussienne, lorentzienne, et sinc
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma):
    return (amplitude * gamma**2) / ((x - center)**2 + gamma**2)

def sinc(x, amplitude, center, width):
    # Calcul de sinc en normalisant (x - center) pour définir la largeur
    return amplitude * np.sinc((x - center) / width)

# Ajuster les données
x = np.arange(len(int))  # Positions des pixels

# Modèle gaussien
gauss_model = Model(gaussian)
gauss_result = gauss_model.fit(int, x=x, amplitude=np.max(int), center=np.argmax(int), sigma=1)
fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * gauss_result.params['sigma'].value

# Modèle lorentzien
lorentz_model = Model(lorentzian)
lorentz_result = lorentz_model.fit(int, x=x, amplitude=np.max(int), center=np.argmax(int), gamma=1)
fwhm_lorentz = 2 * lorentz_result.params['gamma'].value

# Modèle sinc
sinc_model = Model(sinc)
sinc_result = sinc_model.fit(int, x=x, amplitude=np.max(int), center=np.argmax(int), width=5)
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
print(f" - R² : {1 - (np.sum((int - gauss_result.best_fit) ** 2) / np.sum((int - np.mean(int)) ** 2)):.4f}")

# Statistiques pour l'ajustement lorentzien
print("\nStatistiques de l'ajustement lorentzien :")
print(f" - Chi-carré : {lorentz_result.chisqr}")
print(f" - Chi-carré réduit : {lorentz_result.redchi}")
print(f" - R² : {1 - (np.sum((int - lorentz_result.best_fit) ** 2) / np.sum((int - np.mean(int)) ** 2)):.4f}")

# Statistiques pour l'ajustement sinc
print("\nStatistiques de l'ajustement sinc :")
print(f" - Chi-carré : {sinc_result.chisqr}")
print(f" - Chi-carré réduit : {sinc_result.redchi}")
print(f" - R² : {1 - (np.sum((int - sinc_result.best_fit) ** 2) / np.sum((int - np.mean(int)) ** 2)):.4f}")

# Comparer les résultats et déterminer le meilleur ajustement
best_model = min(
    [
        ("gaussien", gauss_result.redchi, gauss_result),
        ("lorentzien", lorentz_result.redchi, lorentz_result),
        ("sinc", sinc_result.redchi, sinc_result),
    ],
    key=lambda item: (abs(item[1] - 1), -1 * (1 - np.sum((int - item[2].best_fit) ** 2) / np.sum((int - np.mean(int)) ** 2)))
)

print("\nComparaison des ajustements :")
print(f"Le meilleur choix pour ajuster les données est le modèle {best_model[0]}.")

# Visualisation des ajustements
plt.plot(x, int, label="Intensité bleue", color="blue")
plt.plot(x, gauss_result.best_fit, 'k--', label=f"Ajustement Gaussien (FWHM = {fwhm_gauss:.2f})")
plt.plot(x, lorentz_result.best_fit, 'g--', label=f"Ajustement Lorentzien (FWHM = {fwhm_lorentz:.2f})")
plt.plot(x, sinc_result.best_fit, 'r--', label=f"Ajustement Sinc (FWHM approx = {fwhm_sinc:.2f})")

# Ajouter des détails de légende et de mise en forme
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe d'intensité et ajustements gaussien, lorentzien et sinc")
plt.legend()
plt.show()
