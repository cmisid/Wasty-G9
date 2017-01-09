# -*- conding = UTF-8 -*-
"""
fonctions de prétraitement de l'image avant l'extraction des
descripteurs

Nous avons dans l'ordre une lecture de l'image 

Un redimensionnement de l'image

Et une transformation en niveau de gris

"""

import os
import skimage
from skimage import io, data, feature, filters
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi
import math


# Fonction permettant de lire une image
def lireImage(nom_im):
    """
    Met en commun la méthode de lecture
    d'image
    """
    return skimage.io.imread(nom_im)

def redimensionner(img, TARGETED_HEIGHT = 100, TARGETED_WIDTH = 100):
    """
    Redimensionner une image

    TARGETED_HEIGHT correspond à la hauteur voulu
    TARGETED_WIDTH correspond à la largeur voulu
    """
    # Taille de départ
    image_height = np.shape(img)[0]
    image_width = np.shape(img)[1]
    # Cote limitant de l'image
    valeur_lim = np.argmin(np.shape(img)[0:2])
    # proportion entre la hauteur et la largeur
    heigt_width_proportion = image_height/image_width

    # Si largeur limitant
    if not valeur_lim:
        # Redimensionnement proportionnel
        img_resized = resize(img, (TARGETED_HEIGHT,\
         math.ceil(TARGETED_WIDTH * 1/heigt_width_proportion)))
        # Découpage de l'image
        width_center = math.ceil(np.shape(img_resized)[1]/2)
        img_resized = img_resized[:, width_center-50:width_center+50]
    # Si hauteur limitant
    else:
        # Redimensionnement proportionnel
        img_resized = resize(img, (math.ceil(TARGETED_HEIGHT*\
        heigt_width_proportion), TARGETED_WIDTH))
        # Découpage
        height_center = math.ceil(np.shape(img_resized)[0]/2)
        img_resized = img_resized[height_center-50:height_center+50, :]

    return img_resized

def conversionRgbToGray(img):
    """
    Met en commun la méthode de conversion de
    l'image
    """
    return rgb2gray(img)

if __name__ == '__main__':
    img = lireImage("image2.jpg")
    img_redim = redimensionner(img)
    img_grey = conversionRgbToGray(img)
    skimage.io.imshow(img)
    skimage.io.imshow(img_redim)
    skimage.io.imshow(img_grey)
    skimage.io.show()
