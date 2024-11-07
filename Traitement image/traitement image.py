import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = Image.open("Photo-lumiere_blanche-lames-3D-testatte1x.tif").convert("L")
image_array = np.array(image)
imagefiltre = Image.open("Spectro-laser_rouge-lames-3D.tif").convert("L")
image_arrayfiltre = np.array(imagefiltre)
image_bleu = Image.open("Spectro-laser_bleu-lames-3D.tif").convert("L")
image_bleu_array = np.array(image_bleu)
image_rouge = Image.open("Spectro-laser_rouge-lames-3D.tif").convert("L")
image_rouge_array = np.array(image_rouge)

def intensite(matrice_image, type_de_spectro="table"):
    if type_de_spectro=="3D":
        matrice_image=np.flip(matrice_image)
    
    # Étape 1 : Trouver l'intensité maximale dans l'image
    intensité_max = np.max(matrice_image)
    if intensité_max == 255:
        print("L'image est saturée")
        return 
    
    # Étape 2 : Identifier les pixels ayant une intensité >= 90% de l'intensité maximale
    seuil = 0.95 * intensité_max
    mask = matrice_image >= seuil

    # Étape 3 : Trouver les lignes contenant au moins un point au-dessus du seuil
    lignes_à_considerer = np.any(mask, axis=1)

    # Étape 4 : Sélectionner ces lignes et calculer la moyenne d'intensité par colonne
    intensité_par_colonne = np.mean(matrice_image[lignes_à_considerer, :], axis=0)

    return intensité_par_colonne


intimage=intensite(image_array)
intfiltre=intensite(image_arrayfiltre,"table")
introuge=intensite(image_rouge_array)
intbleu=intensite(image_bleu_array)

x_rouge=np.argmax(introuge)
x_bleu=np.argmax(intbleu)
echelle_pixel=x_rouge-x_bleu
delta_echelle_pixel=2
echelle_lamda=650-405
delta_lambda_bleu=4
x = np.linspace(0, len(intimage) - 1, len(intimage))

val_lamda=[]
for i in x:
    val_lamda.append((echelle_lamda/echelle_pixel)*(i-x_bleu)+405)

val_lamda=np.array(val_lamda)

# def calcule_incertitude_val_lamda(intimage, echelle_lamda, echelle_pixel, x_bleu, delta_echelle_lamda, delta_echelle_pixel, delta_x_bleu, delta_lambda_bleu):
#     x = np.linspace(0, len(intimage) - 1, len(intimage))
#     val_lamda = []
#     incertitude_val_lamda = []
    
#     for i in x:
#         # Calcul de val_lamda
#         val = (echelle_lamda / echelle_pixel) * (i - x_bleu) + 405
#         val_lamda.append(val)
        
#         # Calcul de l'incertitude
#         term1 = ((i - x_bleu) / echelle_pixel) * delta_echelle_lamda
#         term2 = (-echelle_lamda * (i - x_bleu) / (echelle_pixel ** 2)) * delta_echelle_pixel
#         term3 = (echelle_lamda / echelle_pixel) * delta_x_bleu
#         delta_val = np.sqrt(term1**2 + term2**2 + term3**2)+delta_lambda_bleu
        
#         incertitude_val_lamda.append(delta_val)
    
#     val_lamda = np.array(val_lamda)
#     incertitude_val_lamda = np.array(incertitude_val_lamda)
    
#     return val_lamda, incertitude_val_lamda


# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(val_lamda, intimage)
plt.plot(val_lamda, intfiltre)
plt.xlabel("Position horizontale (nm)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité (lignes sélectionnées)")
plt.show()

def transfert_filter(image_fond, image_filtre, type_de_spectro="table"):
    fond=intensite(image_fond, type_de_spectro)
    filtre=intensite(image_filtre, "table")
    fond_min_non_zero = np.min(fond[fond > 0]) 
    fonction_de_transfert=filtre/(fond+fond_min_non_zero)
    return fonction_de_transfert

fct_trans=transfert_filter(image_array,image_arrayfiltre)
#on de transfert en fonction de la position horizontale

plt.plot(val_lamda, fct_trans)
plt.xlabel("Position horizontale (nm)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité (lignes sélectionnées)")
plt.show()