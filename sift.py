#Groupe [9]
#

import os
import cv2
import json


#Entrée : url d'une image
#Description : prend une image en entrée et retourne le descripteur associé
#Sortie :
def sift_descriptor(image_url):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descripteur = sift.detectAndCompute(image,None)
    image = dict()
    image = {
                 'indice' : i,
                 'category' : None,
                 "desc_sift" : descripteur.tolist()
             }
    category = None
    # classification code

    # ---------
    image['category'] = category
    return image

#Entrée :
#Description :
#Sortie :
def generate_train_descriptors():
    sift = cv2.xfeatures2d.SIFT_create()
    descripteurs = []
    images = []
    i = 0
    #Browsing image files
    for dirname, dirnames, filenames in os.walk("/Users/ismailaddou/Wasty-ImageClassifier/BD_image_pro"):
        for filename in filenames:
            i = i + 1
            path = os.path.join(dirname,filename)
            category = dirname.split('/')
            category = category[len(category)-1]
            img1 = cv2.imread(path,0)
            #kp: keypoints, des: descriptors
            kp, des = sift.detectAndCompute(img1,None)
            image = dict()
            image = {
                         'indice' : i,
                         'category' : category,
                         "desc_sift" : des.tolist()
                     }
            images.append(image)

    #Output file : descriptor.json
    output = json.dumps(images)
    output = json.loads(output)
    with open('descriptors.json', 'w') as outfile:
        json.dump(output, outfile)
