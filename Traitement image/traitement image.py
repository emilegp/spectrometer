import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = Image.open("Photo-laser-blanche-13um-table.tif").convert("L")
image_array = np.array(image)
imagefiltre = Image.open("Photo-laser-rouge-atténué-n0=0.6-5um-table.tif").convert("L")
image_arrayfiltre = np.array(imagefiltre)


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


int=intensite(image_array)
intfiltre=intensite(image_arrayfiltre,"3D")

# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(int)
plt.plot(intfiltre)
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité (lignes sélectionnées)")
plt.show()

def transfert_filter(image_fond, image_filtre, type_de_spectro="table"):
    fond=intensite(image_fond, type_de_spectro)
    filtre=intensite(image_filtre, "3D")

    difference=fond-filtre

    fonction_de_transfert=difference/(np.max(difference)+0.001)
    return fonction_de_transfert

fct_trans=transfert_filter(image_array,image_arrayfiltre)

# Tracer la courbe de la fonction de transfert en fonction de la position horizontale
plt.plot(fct_trans)
plt.xlabel("Position horizontale (pixels)")
plt.ylabel("Intensité moyenne")
plt.title("Courbe de l'intensité (lignes sélectionnées)")
plt.show()