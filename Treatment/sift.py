#Groupe [9]
#

import os
import cv2
import json 
import numpy as np
from pprint import pprint
from operator import itemgetter

##CONSTANTS
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
    #Browsing image files
    for dirname, dirnames, filenames in os.walk("./Image/BD_image_pro"):
        for filename in filenames:
            if(dirname != './Image/BD_image_pro'):
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
                image = {
                             'indice' : i,
                             'category' : category,
                             'desc_sift' : desc_sift
                         }
                images.append(image)
    
    print(len(image))
    #Output file : descriptor.json
    output = json.dumps(images)
    output = json.loads(output)
    with open('descriptors.json', 'w') as outfile:
        json.dump(output, outfile)
    return images

def get_train_descriptor():
    with open('descriptors.json') as data_file:    
        data = json.load(data_file)
        return data
        
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
    return sorted_results    
 
    
    
    
    
    
    
