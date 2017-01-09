#Groupe [9]
#

import os
import cv2
import json
import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter

##CONSTANTS pour le redimmensionnement par défaut des images
HEIGHT = 100
WIDTH = 100
############

#Entrée : une image dans un vecteur numpy
#Description : prend une image en entrée et retourne le descripteur associé
#Sortie : un dictionnaire qui contient l'image en question, la catégorie et les descripteur et les dimensions
def sift_descriptor(image):
    image = cv2.resize(image, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descripteur = sift.detectAndCompute(image,None)
    desc_sift = []
    for idx, val in enumerate(descripteur):
        x,y = keypoints[idx].pt
        desc_sift.append({'kp_x': x , 'kp_y':y , 'descripteur' : val.tolist()})
    image = dict()
    image = {
                 'img_width' : WIDTH,
                 'img_height' : HEIGHT,
                 'category' : None,
                 'desc_sift' : desc_sift.tolist()
             }
    category = None
    image['category'] = category
    return image

#Description : Retraitement de la base d'images pour en faire ressortir des descripteurs
def update_train_descriptors():
    #image = cv2.resize(image, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
    sift = cv2.xfeatures2d.SIFT_create()
    descripteurs = []
    images = []
    i = 0
    errors = []
    #Browsing image files
    for dirname, dirnames, filenames in os.walk("./Image/BD_mini/"):
        for filename in filenames:
            if(dirname != './Image/BD_mini/'):
                try:
                    i = i + 1
                    path = os.path.join(dirname,filename)
                    category = dirname.split('/')
                    category = category[len(category)-1]
                    img1 = cv2.imread(path,0)
                    #kp: keypoints, des: descriptors
                    kp, des = sift.detectAndCompute(img1,None)
                    desc_sift = []
                    for idx, val in enumerate(des):
                        x,y = kp[idx].pt
                        desc_sift.append({'kp_x': x , 'kp_y':y , 'descripteur' : val.tolist()})
                    image = dict()
                    image = { 'indice' : i, 'category' : category, 'desc_sift' : desc_sift  }
                    images.append(image)
                    print(i)
                except:
                    errors.append(path)
    print(len(images))
    print(len(errors))
    #Output file : descriptor.json
    output = json.dumps(images)
    output = json.loads(output)
    with open('descriptors.json', 'w') as outfile:
        json.dump(output, outfile)

#Description : Fonction utilisé en interne, pour charger le fichiers JSON des descripteurs
#Sortie: Descripteur en sortie
def get_train_descriptor():
    with open('descriptors.json') as data_file:
        data = json.load(data_file)
        return data

#Entrée : numpy_img => une image dans un vecteur numpy
#         precision => La précision du matching avec la méthode sift (recommandation: precision>0.7)
#Description : prend une image en entrée et retourne le descripteur associé
#Sortie : une liste ordonnée de prédictions, la première case correspond à la prédition la plus probable
def predict_class(numpy_img,precision):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(numpy_img,None)
    images = []
    data = get_train_descriptor()
    for item in data:
        des = []
        for subitem in item['desc_sift']:
            #d = np.array(subitem['descripteur'])
            d = subitem['descripteur']
            des.append(d)
            des2 = np.array(des,dtype = np.float32)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < precision *n.distance:
                good.append([m])

        image = dict()
        image = {
                     'indice' : item['indice'],
                     'category' : item['category'],
                     'nb_matches' : len(good)
                 }
        images.append(image)
    sorted_results = sorted(images, key=itemgetter('nb_matches'), reverse=True)
    return compute(sorted_results)

def compute(sorted_results):
    df = pd.DataFrame(sorted_results)
    means = df.groupby('category').mean()
    category = means[means['nb_matches'] == means.nb_matches.max()].index.tolist()
    sorted_list = means.sort_index(by=['nb_matches'],ascending=[False]).index.tolist()
    return sorted_list


def bag_of_words_descriptor():
#1. Obtain the set of bags of features.
#   Select a large set of images.
#   Extract the SIFT feature points of all the images in the set and obtain the SIFT descriptor for each feature point that is extracted from each image.
#   Cluster the set of feature descriptors for the amount of bags we defined and train the bags with clustered feature descriptors (we can use the K-Means algorithm).
#   Obtain the visual vocabulary.
#2. Obtain the BoF descriptor for given image/video frame.
#   Extract SIFT feature points of the given image.
#   Obtain SIFT descriptor for each feature point.
#   Match the feature descriptors with the vocabulary we created in the first step
#   Build the histogram.
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))

    # List where all the descriptors are stored
    des_list = []

    for image_path in image_paths:
        im = cv2.imread(image_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))
