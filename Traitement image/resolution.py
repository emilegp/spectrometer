import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lmfit import Model

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


def calcule_incertitude_val_lamda(rouge,bleu):
    x_rouge=np.argmax(rouge)
    x_bleu=np.argmax(bleu)
    echelle_pixel=x_rouge-x_bleu
    delta_echelle_pixel=2
    echelle_lamda=657-405
    delta_lambda_bleu=5
    delta_echelle_lambda=10
    x = np.linspace(0, len(rouge) - 1, len(rouge))
   
    val_lamda = []
    incertitude_val_lamda = []
    
    for i in x:
        # Calcul de val_lamda
        val = (echelle_lamda / echelle_pixel) * (i - x_bleu) + 405
        val_lamda.append(val)
        # Calcul de l'incertitude
        term1 = ((i - x_bleu) / echelle_pixel) * delta_echelle_lambda
        term2 = (-echelle_lamda * (i - x_bleu) / (echelle_pixel ** 2)) * delta_echelle_pixel
        term3 = (echelle_lamda / echelle_pixel) * delta_echelle_pixel
        delta_val = np.sqrt(term1**2 + term2**2 + term3**2)+delta_lambda_bleu

        incertitude_val_lamda.append(delta_val)
    
    val_lamda = np.array(val_lamda)
    incertitude_val_lamda = np.array(incertitude_val_lamda)
   
    return val_lamda, incertitude_val_lamda

# Calculer les longueurs d'onde et leurs incertitudes
val_lamda, incertitude_val_lamda = calcule_incertitude_val_lamda(introuge, intbleu)

print(val_lamda)


# Tracer l'évolution de l'incertitude de la longueur d'onde en fonction de la longueur d'onde
plt.figure(figsize=(10, 6))
plt.plot(val_lamda, incertitude_val_lamda, 'g-', label="Incertitude sur la longueur d'onde")

# Ajouter des labels et un titre
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Incertitude de la longueur d'onde (nm)")
plt.title("Évolution de l'incertitude de la longueur d'onde en fonction de la longueur d'onde")
plt.legend()
plt.grid(True)
plt.show()




# Définir les modèles de fonctions gaussienne, lorentzienne, et sinc**2
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma):
    return (amplitude * gamma**2) / ((x - center)**2 + gamma**2)

def sinc_squared(x, amplitude, center, width):
    # Calcul de sinc**2 en normalisant (x - center) pour définir la largeur
    return amplitude * (np.sinc((x - center) / width))**2

# Ajuster les données
x = np.arange(len(introuge))  # Positions des pixels

# Modèle gaussien
gauss_model = Model(gaussian)
gauss_result = gauss_model.fit(introuge, x=x, amplitude=np.max(introuge), center=np.argmax(introuge), sigma=1)
fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * gauss_result.params['sigma'].value

# Modèle lorentzien
lorentz_model = Model(lorentzian)
lorentz_result = lorentz_model.fit(introuge, x=x, amplitude=np.max(introuge), center=np.argmax(introuge), gamma=1)
fwhm_lorentz = 2 * lorentz_result.params['gamma'].value

# Modèle sinc**2
sinc_model = Model(sinc_squared)
sinc_result = sinc_model.fit(introuge, x=x, amplitude=np.max(introuge), center=np.argmax(introuge), width=0.5)
# La FWHM d'une fonction sinc**2 n'est pas aussi facilement définie que pour les gaussiennes et lorentziennes
# mais on pourrait estimer la largeur à mi-hauteur en fonction du paramètre `width`
fwhm_sinc = 2 * sinc_result.params['width'].value  # Approximation pour FWHM de sinc**2

# Imprimer les résolutions
print(f"Résolution gaussienne (FWHM) : {fwhm_gauss:.2f} pixels")
print(f"Résolution lorentzienne (FWHM) : {fwhm_lorentz:.2f} pixels")
print(f"Résolution sinc**2 approximée (FWHM) : {fwhm_sinc:.2f} pixels")

# Afficher les statistiques pour chaque ajustement

# Statistiques pour l'ajustement gaussien
print("\nStatistiques de l'ajustement gaussien :")
print(f" - Chi-carré : {gauss_result.chisqr}")
print(f" - Chi-carré réduit : {gauss_result.redchi}")
print(f" - R² : {1 - (np.sum((introuge - gauss_result.best_fit) ** 2) / np.sum((introuge - np.mean(introuge)) ** 2)):.4f}")

# Statistiques pour l'ajustement lorentzien
print("\nStatistiques de l'ajustement lorentzien :")
print(f" - Chi-carré : {lorentz_result.chisqr}")
print(f" - Chi-carré réduit : {lorentz_result.redchi}")
print(f" - R² : {1 - (np.sum((introuge - lorentz_result.best_fit) ** 2) / np.sum((introuge - np.mean(introuge)) ** 2)):.4f}")

# Statistiques pour l'ajustement sinc**2
print("\nStatistiques de l'ajustement sinc**2 :")
print(f" - Chi-carré : {sinc_result.chisqr}")
print(f" - Chi-carré réduit : {sinc_result.redchi}")
print(f" - R² : {1 - (np.sum((introuge - sinc_result.best_fit) ** 2) / np.sum((introuge - np.mean(introuge)) ** 2)):.4f}")

# Comparer les résultats et déterminer le meilleur ajustement
best_model = min(
    [
        ("gaussien", gauss_result.redchi, gauss_result),
        ("lorentzien", lorentz_result.redchi, lorentz_result),
        ("sinc**2", sinc_result.redchi, sinc_result),
    ],
    key=lambda item: (abs(item[1] - 1), -1 * (1 - np.sum((introuge - item[2].best_fit) ** 2) / np.sum((introuge - np.mean(introuge)) ** 2)))
)

print("\nComparaison des ajustements :")
print(f"Le meilleur choix pour ajuster les données est le modèle {best_model[0]}.")

# Fonction pour convertir une position de pixel en longueur d'onde
def pixel_to_lambda(pixel_position, echelle_lamda, echelle_pixel, x_bleu):
    return (echelle_lamda / echelle_pixel) * (pixel_position - x_bleu) + 405

# Trouver les positions de largeur à mi-hauteur en pixels pour chaque modèle
def fwhm_in_lambda(model_result, echelle_lamda, echelle_pixel, x_bleu):
    center_pixel = model_result.params['center'].value
    fwhm_pixel = 2 * np.sqrt(2 * np.log(2)) * model_result.params['sigma'].value if 'sigma' in model_result.params else \
                 2 * model_result.params['gamma'].value if 'gamma' in model_result.params else \
                 2 * model_result.params['width'].value

    # Calculer les positions de pixels à mi-hauteur autour du centre
    pixel_left = center_pixel - fwhm_pixel / 2
    pixel_right = center_pixel + fwhm_pixel / 2

    # Convertir les positions de pixels en longueurs d'onde
    lambda_left = pixel_to_lambda(pixel_left, echelle_lamda, echelle_pixel, x_bleu)
    lambda_right = pixel_to_lambda(pixel_right, echelle_lamda, echelle_pixel, x_bleu)

    # Calculer la FWHM en longueur d'onde
    return abs(lambda_right - lambda_left)

# Paramètres pour la conversion
x_rouge = np.argmax(introuge)
x_bleu = np.argmax(intbleu)
echelle_pixel = x_rouge - x_bleu
echelle_lamda = 657 - 405

# Calcul de la FWHM en unités de longueur d'onde pour chaque modèle
fwhm_gauss_lambda = fwhm_in_lambda(gauss_result, echelle_lamda, echelle_pixel, x_bleu)
fwhm_lorentz_lambda = fwhm_in_lambda(lorentz_result, echelle_lamda, echelle_pixel, x_bleu)
fwhm_sinc_lambda = fwhm_in_lambda(sinc_result, echelle_lamda, echelle_pixel, x_bleu)

# Visualisation des ajustements
plt.plot(val_lamda, intbleu, label="Intensité bleue", color="blue")
# plt.plot(val_lamda, gauss_result.best_fit, 'k--', label=f"Ajustement Gaussien (FWHM = {fwhm_gauss:.2f})")
# plt.plot(val_lamda, lorentz_result.best_fit, 'g--', label=f"Ajustement Lorentzien (FWHM = {fwhm_lorentz:.2f})")
# plt.plot(val_lamda, sinc_result.best_fit, 'r--', label=f"Ajustement Sinc**2 (FWHM approx = {fwhm_sinc:.2f})")

# Ajouter des détails de légende et de mise en forme
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe d'intensité et ajustements gaussien, lorentzien et sinc**2")
plt.legend()
plt.show()

# Affichage des résultats
print(f"Résolution gaussienne (FWHM) : {fwhm_gauss_lambda:.2f} nm")
print(f"Résolution lorentzienne (FWHM) : {fwhm_lorentz_lambda:.2f} nm")
print(f"Résolution sinc**2 approximée (FWHM) : {fwhm_sinc_lambda:.2f} nm")

# Visualisation des ajustements avec les largeurs à mi-hauteur en unités de longueur d'onde
plt.plot(val_lamda, introuge, label="Intensité rouge", color="red")
plt.plot(val_lamda, gauss_result.best_fit, 'k--', label=f"Ajustement Gaussien (FWHM = {fwhm_gauss_lambda:.2f} nm)")
plt.plot(val_lamda, lorentz_result.best_fit, 'g--', label=f"Ajustement Lorentzien (FWHM = {fwhm_lorentz_lambda:.2f} nm)")
plt.plot(val_lamda, sinc_result.best_fit, 'b--', label=f"Ajustement Sinc**2 (FWHM approx = {fwhm_sinc_lambda:.2f} nm)")

# Ajouter des détails de légende et de mise en forme
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe d'intensité et ajustements gaussien, lorentzien et sinc**2 (en nm)")
plt.legend()
plt.grid(True)
plt.show()
