# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:00:03 2017
Programme qui permet de faire ressortir les contours d´une image
en extrayant la partie interessante de l´image
@author: Groupe 9: Classification d´images
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, color
def Canny__Contours_right(img):
    """ Fonction qui prend en entree une image
        Trace les contours de l´image et la renvoie """
    # img=tr.resize(img,(100,100))
    img_gray = color.rgb2gray(img)
    edges = feature.canny(img_gray, sigma=3)
    haut, lar = edges . shape
    i = 0
    j = 0
    for i in range(haut):
        j = 0
        while not edges[i, j] and j < lar-1:
            j += 1
        if edges[i, j]:
            edges[i, j:] = True
    return edges
def Canny__Contours_left(img):
    """Fonction qui prend en entree une image, qui la parcourt de gauche a droite
        et qui renvoie une partie de l´image"""
    # img=tr.resize(img,(100,100))
    img_gray = color.rgb2gray(img)
    edges = feature.canny(img_gray, sigma=3)
    haut, lar = edges . shape
    i = 0
    j = 1
    for i in range(haut):
        j = 0
        while not edges[i, j] and j > -lar:
            j -= 1
        if edges[i, j]:
            edges[i, :j] = True
    return edges
def Canny__Contours_upper(img):
    """Fonction qui prend en entree une image, qui la parcourt de droite a gauche
        et qui renvoie une partie de l´image"""
    # img=tr.resize(img,(100,100))
    img_gray = color.rgb2gray(img)
    edges = feature.canny(img_gray, sigma=3)
    haut, lar = edges.shape
    i = 0
    j = 1
    for j in range(lar):
        i = 0
        while not edges[i, j] and i > -haut:
            i -= 1
        if edges[i, j]:
            edges[:i, j] = True
    return edges
def Canny__Contours_lower(img):
    """Fonction qui prend en entree une image, qui la parcourt du haut vers le bas
       et qui renvoie une partie de l´image"""
    # img=tr.resize(img,(100,100))
    img_gray = color.rgb2gray(img)
    edges = feature.canny(img_gray, sigma=3)
    haut, lar = edges.shape
    i = 0
    j = 0
    for j in range(lar):
        i = 0
        while not edges[i, j] and i < haut-1:
            i += 1
        if edges[i, j]:
            edges[i:, j] = True
    return edges
def Canny_findContours(image):
    """Fonction qui prend en entree une image, qui utilise les fonctions
    precedentes pour reconstituer l'image finale"""
    edges1 = Canny__Contours_right(image)
    edges2 = Canny__Contours_left(image)
    edges3 = Canny__Contours_lower(image)
    edges4 = Canny__Contours_upper(image)
    h, l, c = np.shape(image)
    for i in range(h):
        for j in range(l):
            if not edges1[i, j] or not edges2[i, j] or not edges3[i, j] or not edges4[i, j]:
                image[i, j, :] = 0
    return image

if __name__=='__main__':
# Test des fonctions
    img = plt.imread('C:/Users/samba/.spyder-py3/23.jpg')
    c = Canny_findContours(img)
    io.imshow(c)
