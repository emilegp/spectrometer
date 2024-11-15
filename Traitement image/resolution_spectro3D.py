#Code pour chaque spectromètre

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lmfit import Model
#Partie1: Image en intensités linéaire
# 1. Tif valeurs de pixel de 0 à 255, donc intensité directement. Permet aussi de vérifier la saturation
# 2. Pour éviter les sections totalement nulles, on sélectionne les lignes pour lesquelles au moins un pixel à une valeur d'au moins 95% de l'intensité maximale
# 2.5 Pour chaque colonne, on fait la moyenne des éléments sur les lignes selectionnées, ce qui retourne l'intensité en fonction de la position horizontale. 
# À Noter: Vérification que l'image n'est pas saturée
# Fait pour rouge et bleu

# Charger les images en niveaux de gris
image_bleu = Image.open("Spectro-laser_bleu-lames-3D.tif").convert("L")
image_bleu_array = np.array(image_bleu)
image_rouge = Image.open("Spectro-laser_rouge-lames-3D.tif").convert("L")
image_rouge_array = np.array(image_rouge)

def intensite(matrice_image):
    intensité_max = np.max(matrice_image)
    if intensité_max == 255:
        print("L'image est saturée")
        return

    seuil = 0.5 * intensité_max
    mask = matrice_image >= seuil
    lignes_à_considerer = np.any(mask, axis=1)
    intensité_par_colonne = np.mean(matrice_image[lignes_à_considerer, :], axis=0)

    return intensité_par_colonne

# Extraire les courbes d'intensité
introuge = intensite(image_rouge_array)
intbleu = intensite(image_bleu_array)

#Partie 2: Conversion de px en lambda 
# 1. Formule de conversion de px en lambda (comment obtenue et forme)
# 2. Propagation d'incertitude sur la conversion avec méthode des dérivées. Mentionner laquelle est la plus importante, Obtention de chacune des incertitudes utilisées
# 3. On obtient la position des pics en lambda et l'incertitude sur cette position. Ce qui permet d'avoir l'intensité pour chaque longueur d'onde

def calcule_incertitude_val_lamda(rouge,bleu):
    x_rouge=np.argmax(rouge)
    x_bleu=np.argmax(bleu)
    echelle_pixel=x_rouge-x_bleu
    delta_echelle_pixel=2 #On considère que c'est l'addition de 2 incertitudes. Chacune étant la plus petite division, soit 1 px.
    echelle_lamda=657-405
    delta_lambda_bleu=(echelle_lamda/echelle_pixel)*delta_echelle_pixel #On reporte l'incertitude des pixel sur les longueurs d'onde
    delta_echelle_lambda=2*delta_lambda_bleu
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
        #print(f'terme1:{term1}, terme2:{term2}, term3:{term3}')
        incertitude_val_lamda.append(delta_val)
    
    val_lamda = np.array(val_lamda)
    incertitude_val_lamda = np.array(incertitude_val_lamda)
   
    return val_lamda, incertitude_val_lamda

# Calculer les longueurs d'onde et leurs incertitudes
val_lamda, incertitude_val_lamda = calcule_incertitude_val_lamda(introuge, intbleu)


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

# Partie 3: Fits
# 1. Présenter les fits testés: Sinc^2 par les maths, gaussienne par habitude et lorentienne par allure. Paramètres et Latex 
# 2. Méthode de moindre carré
# 3. Statistiques pour déterminer lequel est meilleur. TABLEAU. Chi-carré et R^2, et résolution obtenue
# 4. Affichage des données et des fits ensemble pour visualiser. METTRE GRAPH dans doc

#Partie 3: Largeur à mi-hauteur
# 1. Technique utilisée pour obtenir la largeur (maximum, positions à la moitié du max, un peu overshoot au cas où)
# 2. Comparaison entre les 3 spectromètres.

def calculer_FWHM(x, y):
    # 1. Identifier la valeur maximale de l'intensité
    I_max = np.max(y)
    
    # 2. Calculer la moitié de cette valeur
    I_half = I_max / 2
    
    # 3. Trouver les indices où l'intensité atteint la moitié du maximum
    indices_fwhm = np.where(y >= I_half)[0]
    
    # 4. Détermination des bornes supérieure et inférieure en étant sûr de ne pas en manquer (incertitude)
    if I_half <= x[indices_fwhm[-1]]:
        x_fwhm_max = x[indices_fwhm[-1]+1]
    else:
        x_fwhm_max = x[indices_fwhm[-1]]

    if I_half <= x[indices_fwhm[0]]:
        x_fwhm_min = x[indices_fwhm[0]-1]
    else:
        x_fwhm_min = x[indices_fwhm[0]]

    # 5. La FWHM est la distance entre les deux indices extrêmes
    fwhm = x_fwhm_max - x_fwhm_min

    return fwhm, x_fwhm_min, x_fwhm_max, I_half

# Exemple d'utilisation avec les données d'intensité rouge (introuge) et de longueur d'onde (val_lamda)
fwhm, x_fwhm_min, x_fwhm_max, I_half = calculer_FWHM(val_lamda, introuge)
fwhmbleu, x_fwhm_minbleu, x_fwhm_maxbleu, I_half_bleu = calculer_FWHM(val_lamda, intbleu)

print(f"FWHM_rouge: {fwhm} nm, à partir de {x_fwhm_min} nm à {x_fwhm_max} nm")
print(f"FWHM_bleu: {fwhmbleu} nm, à partir de {x_fwhm_minbleu} nm à {x_fwhm_maxbleu} nm")

# Visualisation avec les lignes représentant la FWHM
plt.plot(val_lamda, introuge, label="Intensité rouge", color="red")
plt.axvline(x=x_fwhm_min, color='black', linestyle='--', label="Limite inférieure FWHM")
plt.axvline(x=x_fwhm_max, color='black', linestyle='--', label="Limite supérieure FWHM")
plt.axhline(y=I_half, color='blue', linestyle='--', label="Moitié de la hauteur maximale")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité")
plt.title("Calcul de la FWHM sur la courbe d'intensité")
plt.legend()
plt.grid(True)
plt.show()

# Visualisation avec les lignes représentant la FWHM
plt.plot(val_lamda, intbleu, label="Intensité rouge", color="blue")
plt.axvline(x=x_fwhm_minbleu, color='black', linestyle='--', label="Limite inférieure FWHM")
plt.axvline(x=x_fwhm_maxbleu, color='black', linestyle='--', label="Limite supérieure FWHM")
plt.axhline(y=I_half_bleu, color='blue', linestyle='--', label="Moitié de la hauteur maximale")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité")
plt.title("Calcul de la FWHM sur la courbe d'intensité")
plt.legend()
plt.grid(True)
plt.show()


# Partie 4: Conclusion
# Comparer les deux spectro (table et 3D) au spectro de Guillaume. Donc Parties 1 à 3 pour 2 spectros. et comparaison avec rouge Guigui
# Différence dans les résolutions obtenues voir comparé avec Guigui
# Incertitudes respectives et dire si elles sont bien/logique ou pas
#  
# 

