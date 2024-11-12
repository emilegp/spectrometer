import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = Image.open("11-11-2024-Spectro-LED-froide-blanche-3D.tif").convert("L")
froid_array = np.array(image)
imagefiltre = Image.open("11-11-2024-Spectro-filtre-LED-froide-blanche-3D.tif").convert("L")
froid_arrayfiltre = np.array(imagefiltre)
image = Image.open("11-11-2024-Spectro-LED-chaude-blanche-3D.tif").convert("L")
chaud_array = np.array(image)
imagefiltre = Image.open("11-11-2024-Spectro-filtre-LED-chaude-blanche-3D.tif").convert("L")
chaud_arrayfiltre = np.array(imagefiltre)
image = Image.open("11-11-2024-Spectro-baton-lumiere-blanche-3D.tif").convert("L")
baton_array = np.array(image)
imagefiltre = Image.open("11-11-2024-Spectro-filtre-baton-lumiere-blanche-3D.tif").convert("L")
baton_arrayfiltre = np.array(imagefiltre)
image_bleu = Image.open("11-11-2024-Spectro-laser_bleu-lames-3D.tif").convert("L")
image_bleu_array = np.array(image_bleu)
image_rouge = Image.open("11-11-2024-Spectro-laser_rouge-lames-3D.tif").convert("L")
image_rouge_array = np.array(image_rouge)
image = Image.open("11-11-2024-Spectro-DEL-rousse-3D.tif").convert("L")
rousse_array = np.array(image)

def intensite(matrice_image, type_de_spectro="3D"):
    if type_de_spectro=="table":
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

introusse=intensite(rousse_array)
intfroid=intensite(froid_array)
intfroidfiltre=intensite(froid_arrayfiltre)
intchaud=intensite(chaud_array)
intchaudfiltre=intensite(chaud_arrayfiltre)
intbaton=intensite(baton_array)
intbatonfiltre=intensite(baton_arrayfiltre)
introuge=intensite(image_rouge_array)
intbleu=intensite(image_bleu_array)

x_rouge=np.argmax(introuge)
x_bleu=np.argmax(intbleu)
echelle_pixel=x_rouge-x_bleu
delta_echelle_pixel=2
echelle_lamda=650-405
delta_lambda_bleu=4
x = np.linspace(0, len(intfroid) - 1, len(intfroid))

val_lamda=[]
for i in x:
    val_lamda.append((echelle_lamda/echelle_pixel)*(i-x_bleu)+405)

val_lamda=np.array(val_lamda)

def calcule_incertitude_val_lamda(intimage, echelle_lamda, echelle_pixel, x_bleu, delta_echelle_lamda, delta_echelle_pixel, delta_x_bleu, delta_lambda_bleu):
    x = np.linspace(0, len(intimage) - 1, len(intimage))
    val_lamda = []
    incertitude_val_lamda = []
    
    for i in x:
        # Calcul de val_lamda
        val = (echelle_lamda / echelle_pixel) * (i - x_bleu) + 405
        val_lamda.append(val)
        
        # Calcul de l'incertitude
        term1 = ((i - x_bleu) / echelle_pixel) * delta_echelle_lamda
        term2 = (-echelle_lamda * (i - x_bleu) / (echelle_pixel ** 2)) * delta_echelle_pixel
        term3 = (echelle_lamda / echelle_pixel) * delta_x_bleu
        delta_val = np.sqrt(term1**2 + term2**2 + term3**2)+delta_lambda_bleu
        
        incertitude_val_lamda.append(delta_val)
    
    val_lamda = np.array(val_lamda)
    incertitude_val_lamda = np.array(incertitude_val_lamda)
    
    return val_lamda, incertitude_val_lamda

# Tracer la courbe de l'intensité en fonction de la position horizontale
plt.plot(val_lamda, introusse/np.max(introusse))
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité normalisée")
plt.title("Intensité de la DEL orange obtenue au spectromètre imprimé en 3D")
plt.show()

# # Tracer la courbe de l'intensité en fonction de la position horizontale
# plt.plot(val_lamda, intfroid)
# plt.plot(val_lamda, intfroidfiltre)
# plt.xlabel("Position horizontale (nm)")
# plt.ylabel("Intensité moyenne")
# plt.title("Courbe de l'intensité (froid)")
# plt.show()

# # Tracer la courbe de l'intensité en fonction de la position horizontale
# plt.plot(val_lamda, intchaud)
# plt.plot(val_lamda, intchaudfiltre)
# plt.xlabel("Position horizontale (nm)")
# plt.ylabel("Intensité moyenne")
# plt.title("Courbe de l'intensité (chaud)")
# plt.show()

# # Tracer la courbe de l'intensité en fonction de la position horizontale
# plt.plot(val_lamda, intbaton)
# plt.plot(val_lamda, intbatonfiltre)
# plt.xlabel("Position horizontale (nm)")
# plt.ylabel("Intensité moyenne")
# plt.title("Courbe de l'intensité (baton))")
# plt.show()

def transfert_filter(image_fond, image_filtre, type_de_spectro="3D"):
    fond=intensite(image_fond, type_de_spectro)
    filtre=intensite(image_filtre, "3D")
    fond_min_non_zero = np.min(fond[fond > 0])/100
    fonction_de_transfert=filtre/(fond+fond_min_non_zero)
    fct_de_transfert_normalisee=fonction_de_transfert/np.max(fonction_de_transfert)
    return fct_de_transfert_normalisee

fct_trans_froid=transfert_filter(froid_array,froid_arrayfiltre)
fct_trans_chaud=transfert_filter(chaud_array,chaud_arrayfiltre)
fct_trans_baton=transfert_filter(baton_array,baton_arrayfiltre)

#Fonction de transfert en fonction de la position horizontale

# plt.plot(val_lamda, fct_trans_baton, label='baton')
# plt.plot(val_lamda, fct_trans_chaud, label= 'chaude')
plt.plot(val_lamda, fct_trans_froid)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Fonction de transfert normalisée")
plt.title("Fonction de transfert de la LED froide")
plt.show()