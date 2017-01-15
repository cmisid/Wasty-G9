# -*- conding = UTF-8 -*-

"""
Groupe9: Classification d'images

Indexation des images grâce aux couleurs

Version 2
"""


import os
from scipy.misc import imread
import numpy as np
from scipy.ndimage.measurements import histogram
from skimage.transform import resize
import pretraitement as prt

# Reduction de la couleur sur 6 bits
# Prend chaque pixel et renvoi sa valeur réduite
def sample_pixel(pixel):
    """
    Reduction de couleurs
    """
    to_bits = lambda x: '{0:08b}'.format(x)
    red, green, blue = to_bits(pixel[0]), to_bits(pixel[1]), to_bits(pixel[2])
    red_intensity = int(red[0]) * 2 ** 5 + int(red[1]) * 2 ** 4
    green_intensity = int(green[0]) * 2 ** 3 + int(green[1]) * 2 ** 2
    blue_intensity = int(blue[0]) * 2 ** 1 + int(blue[1]) * 2 ** 0
    intensity = red_intensity + green_intensity + blue_intensity
    return[intensity, intensity, intensity]
    # Nécessaire d'en mettre 3 pour l'affichage

# prend chaque image et envoi les pixels a la reduction
# retourne l'image reduite
def sample_image(image):
    """
    Echantillonage des pixels
    """
    return [
        [
            sample_pixel(pixel)
            for pixel in row
        ]
        for row in image
        ]


# variable recuperant l'ensemble des données a récupérer dans le csv
text = 'categorie, numim, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63'
path = "/Users/Tom/Desktop/Projet/Wasty-G9/Image/BD_simple/"
# recupération des noms des dossiers de la BD

# boucle pour passer dans tous les dossiers ayant des images (try)
for categorie in os.listdir(path):
    try:
        # chemin de l'image
        chm = os.path.join(path, categorie)
        # boucle pour lire chaque image
        for picture in os.listdir(chm):
            # on lit, puis on redimensionne l'image
            img = imread(os.path.join(chm, picture))
            small_img = prt.Canny_findContours(img)
            # small_img = resize(img, (100, 100))
            small_img = np.ceil(small_img*255)
            small_img = img2.astype(np.int16)
            # on met en nuance de gris
            sample = sample_image(small_img)
            # on recupere l'histogramme
            hist = histogram(np.array(sample)[:, :, 0], min=0, max=63, bins=64)
            num_img = picture.split('.')
            # remplassage du numero de photo et des valeurs de l'histogramme
            text = text + "\n" + categorie + ',' + num_img[0]

            for i in hist:
                text = text + ",{}".format(i)

# exception au cas ou un dossier de contient aucune image
    except NotADirectoryError:
        print("")

# permis de remplir le fichier csv avec texte
fichier = open("/Users/Tom/Desktop/Projet/programme_fin/test2.csv", "w")
fichier.write(text)
fichier.close()
