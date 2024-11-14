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




# Charger les images en niveaux de gris
image_bleu = Image.open("11-11-2024-Spectro-laser_bleu-lames-3D.tif").convert("L")
image_bleu_array = np.array(image_bleu)
image_rouge = Image.open("11-11-2024-Spectro-laser_rouge-lames-3D.tif").convert("L")
image_rouge_array = np.array(image_rouge)

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
introuge = intensite(image_rouge_array)
intbleu = intensite(image_bleu_array)


# Définir les modèles de fonctions gaussienne, lorentzienne, et sinc^2
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma):
    return (amplitude * gamma**2) / ((x - center)**2 + gamma**2)

def sinc_squared(x, amplitude, center, width):
    # Sinc^2 sans normalisation par pi
    return amplitude * (np.sinc((x - center) / width))**2

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

# Modèle sinc**2

# Estimer une largeur initiale pour sinc^2 (par exemple, basée sur l'écart entre les points maximaux et minimaux)
width_init = (np.argmax(intensites) - np.argmin(intensites)) / 10  # Ajustez selon vos données

# Ajustement du modèle sinc^2 avec une largeur initiale estimée
sinc_model = Model(sinc_squared)
sinc_result = sinc_model.fit(intensites, x=x, amplitude=np.max(intensites), center=np.argmax(intensites), width=width_init)

# La FWHM d'une fonction sinc**2 n'est pas aussi facilement définie que pour les gaussiennes et lorentziennes,
# mais on peut l'estimer en fonction du paramètre 'width' du modèle sinc.
fwhm_sinc = 2 * sinc_result.params['width'].value  # Approximation pour FWHM de sinc**2

# Imprimer les résolutions
print(f"Résolution gaussienne (FWHM) : {fwhm_gauss:.2f} pixels")
print(f"Résolution lorentzienne (FWHM) : {fwhm_lorentz:.2f} pixels")
print(f"Résolution sinc^2 approximée (FWHM) : {fwhm_sinc:.2f} pixels")

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

# Statistiques pour l'ajustement sinc^2
print("\nStatistiques de l'ajustement sinc^2 :")
print(f" - Chi-carré : {sinc_result.chisqr}")
print(f" - Chi-carré réduit : {sinc_result.redchi}")
print(f" - R² : {1 - (np.sum((intensites - sinc_result.best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)):.4f}")

# Comparer les résultats et déterminer le meilleur ajustement
best_model = min(
    [
        ("gaussien", gauss_result.redchi, gauss_result),
        ("lorentzien", lorentz_result.redchi, lorentz_result),
        ("sinc^2", sinc_result.redchi, sinc_result),
    ],
    key=lambda item: (abs(item[1] - 1), -1 * (1 - np.sum((intensites - item[2].best_fit) ** 2) / np.sum((intensites - np.mean(intensites)) ** 2)))
)

print("\nComparaison des ajustements :")
print(f"Le meilleur choix pour ajuster les données est le modèle {best_model[0]}.")

# Visualisation des ajustements
plt.plot(x, intensites, label="Intensité", color="blue")
plt.plot(x, gauss_result.best_fit, 'k--', label=f"Ajustement Gaussien (FWHM = {fwhm_gauss:.2f})")
plt.plot(x, lorentz_result.best_fit, 'g--', label=f"Ajustement Lorentzien (FWHM = {fwhm_lorentz:.2f})")
plt.plot(x, sinc_result.best_fit, 'r--', label=f"Ajustement Sinc^2 (FWHM approx = {fwhm_sinc:.2f})")

# Ajouter des détails de légende et de mise en forme
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe d'intensité et ajustements gaussien, lorentzien et sinc^2")
plt.legend()
plt.show()
