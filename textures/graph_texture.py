#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:46:50 2017

@author: Tom
"""

import numpy as np
from scipy.misc import imread
from skimage.transform import resize
import pandas
import pretraitement as prt


def log_mean_energy(array):
    return np.log(np.mean(np.square(array)))


def descriptor(image, n_blocks=(5, 10)):
    grays = np.zeros(image.shape)
    # Moyennage de chaque pixel
    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            grays[i][j] = np.mean(pixel)
    # Extraction de la valeur réelle (avec la fonction abs) de la FFT-2D
    transform = np.abs(np.fft.fft2(grays))
    # Découpage en blocs
    # Moitié supérieure
    top = transform[:transform.shape[0] // 2, :]
    x_step = top.shape[0] // n_blocks[0]
    y_step = top.shape[1] // n_blocks[1]
    return {
        log_mean_energy(top[x:x+x_step, y:y+y_step])
        for j, y in enumerate(range(0, top.shape[1], y_step))
        for i, x in enumerate(range(0, top.shape[0], x_step))
    }


# prend en entrer le chemin de l'image et le chemin du csv
# retourn la categorie de l'image la plus proche avec le score et la moyenne des catégorie
def graph_textur(chm_img, chm_csv):
    picture = pandas.read_csv(chm_csv, sep=',')
    img = imread(chm_img)
    small_img = prt.Canny_findContours(img)
    # small_img = resize(img, (100, 100))
    small_img = np.ceil(small_img*255)
    text = str(descriptor(small_img))
    text = text[1:-1]
    text2 = text.split(',')
    rank = np.zeros((10, 4), dtype=np.int16)
    for i in range(10):
        rank[i][1] = 10000
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
    for i in range(len(picture['categorie'])):
        addition = 0
        for j in range(50):
            sum_hist = abs(float(text2[j]) - float(picture.iloc[i][j+2]))
            addition = addition+sum_hist
        k = 0
        while k < 10:
            if addition < float(rank[k, 1]):
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
    tab = np.zeros((5, 3), dtype=np.int16)
    tab2 = tab.astype(str)
    tab2[0, 0] = 'petits'
    tab2[1, 0] = 'materi'
    tab2[2, 0] = 'electr'
    tab2[3, 0] = 'mobili'
    tab2[4, 0] = 'textil'
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
    main_img[0, 1] = '999999'
    for i in range(5):
        if int(tab2[i, 1]) == maxi:
            if float(main_img[0, 1]) > float(tab2[i, 2]):
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
