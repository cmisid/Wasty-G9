"""
Groupe9: Classification d'images

classification d'une image par la couleur

Version 1
"""


import pandas
from scipy.misc import imread
import numpy as np
from scipy.ndimage.measurements import histogram
from skimage.transform import resize
import numpy
import pretraitement as prt

# Reduction de la couleur sur 6 bits
# Prend chaque pixel et renvoi sa valeur réduite
def sample_pixel(pixel):
    to_bits = lambda x: '{0:08b}'.format(x)
    r, g, b = to_bits(pixel[0]), to_bits(pixel[1]), to_bits(pixel[2])
    r_intensity = int(r[0]) * 2 ** 5 + int(r[1]) * 2 ** 4
    g_intensity = int(g[0]) * 2 ** 3 + int(g[1]) * 2 ** 2
    b_intensity = int(b[0]) * 2 ** 1 + int(b[1]) * 2 ** 0
    intensity = r_intensity + g_intensity + b_intensity
    return [intensity, intensity, intensity]
    # Nécessaire d'en mettre 3 pour l'affichage

# prend chaque image et envoi les pixels a la reduction
# retourne l'image reduite
def sample_image(image):
    return [
        [
            sample_pixel(pixel)
            for pixel in row
        ]
        for row in image
    ]

# on lit, puis on redimensionne l'image
# prend le chemin de l'image et celui du csv et retourne les categories
def graph_coulor(chm_img, chm_csv):
    img = imread(chm_img)
    small_img = resize(img, (100, 100))
    small_img = prt.Canny_findContours(img)
    # small_img = np.ceil(small_img*255)
    small_img = small_img.astype(np.int16)
    # on met en nuance de gris
    sample = sample_image(small_img)
    # on recupere l'histogramme
    hist = histogram(np.array(sample)[:, :, 0], min=0, max=63, bins=64)
    # recuperation de l'image
    picture = pandas.read_csv(chm_csv, sep=',')
    # creation du tableau de classement
    rank = numpy.zeros((10, 4), dtype=numpy.int16)
    rank[0][3] = 10
    rank[1][3] = 9
    rank[2][3] = 8
    rank[3][3] = 7
    rank[4][3] = 6
    rank[5][3] = 5
    rank[6][3] = 4
    rank[7][3] = 3
    rank[8][3] = 2
    rank[9][3] = 1
    rank = rank.astype(str)
    # comparaison entre l'histogramme de la photo
    # et celui de chaque photo de la base
    for i in range(len(picture['categorie'])):
        addition = 0
        for j in range(64):
            sum_hist = min(hist[j], picture.iloc[i][j+2])
            addition = addition+sum_hist
        k = 0
        while k < 10:
            if addition > int(rank[k, 1]):
                # mise a jour du classement si changement
                for l in reversed(range(k+1, 10)):
                    rank[l, 0] = rank[l-1, 0]
                    rank[l, 1] = rank[l-1, 1]
                    rank[l, 2] = rank[l-1, 2]
                rank[k, 0] = picture.iloc[i, 0]
                rank[k, 1] = str(addition)
                rank[k, 2] = picture.iloc[i, 1]
                k = 11
            k += 1
    # classement final de la categorie de l'image
    tab = np.zeros((5, 3), dtype=np.int16)
    tab2 = tab.astype(str)
    tab2[0, 0] = 'petits'
    tab2[1, 0] = 'materi'
    tab2[2, 0] = 'electr'
    tab2[3, 0] = 'mobili'
    tab2[4, 0] = 'textil'
    # creation du score en fonction de la possition dans le classement
    for i in range(10):
        if rank[i][0] == 'petits':
            tab2[0, 1] = str(int(tab2[0, 1])+int(rank[i][3]))
        elif rank[i][0] == 'materi':
            tab2[1, 1] = str(int(tab2[1, 1])+int(rank[i][3]))
        elif rank[i][0] == 'electr':
            tab2[2, 1] = str(int(tab2[2, 1])+int(rank[i][3]))
        elif rank[i][0] == 'mobili':
            tab2[3, 1] = str(int(tab2[3, 1])+int(rank[i][3]))
        elif rank[i][0] == 'textil':
            tab2[4, 1] = str(int(tab2[4, 1])+int(rank[i][3]))
    # calcul du nombre de fois que chaque categorie est presente
    avg = 0
    for i in range(5):
        for j in range(10):
            if tab2[i, 0] == rank[j, 0]:
                avg = avg + float(rank[j, 1])
        if int(tab2[i, 1]) != 0:
            avg = avg / int(tab2[i, 1])
        else:
            tab2[i, 2] = 0
        tab2[i, 2] = avg
        avg = 0
    maxi = 0
    for i in range(5):
        if maxi < int(tab2[i, 1]):
            maxi = int(tab2[i, 1])
    main_img = np.zeros((1, 2), dtype=np.int16)
    main_img = main_img.astype(str)
    main_img[0, 1] = '0'
    for i in range(5):
        if int(tab2[i, 1]) == maxi:
            if float(main_img[0, 1]) < float(tab2[i, 2]):
                main_img[0, 0] = tab2[i, 0]
                main_img[0, 1] = tab2[i, 1]
    category = ''
    best_category = rank[0, 0]
    score = main_img[0][1]
    if main_img[0, 0] == 'petits':
        category = 'petits electromenagers'
    elif main_img[0, 0] == 'materi':
        category = 'materiaux'
    elif main_img[0, 0] == 'electr':
        category = 'electromenager'
    elif main_img[0, 0] == 'mobili':
        category = 'mobiliers'
    elif main_img[0, 0] == 'textil':
        category = 'textiles'
    return category, score, best_category
