#Groupe [9]

import os
import cv2
import json
import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter
from sklearn import neighbors

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

def path(cls,i): # "./left03.jpg"
  return "%s/%s%02d.jpg"  % (datapath,cls,i+1)

def feature_sift(fn,detect,extract):
  im = cv2.imread(fn,0)
  im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
  return extract.compute(im, detect.detect(im))[1]

def feature_bow(fn,detect,bow_extract):
  im = cv2.imread(fn,0)
  im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
  return bow_extract.compute(im, detect.detect(im))

def feature_bow2(im,detect,bow_extract):
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
    #print('length of my image heeeeeeey',image_gray.shape)
    return bow_extract.compute(image_gray, detect.detect(image_gray))

def bof_train_extract_features():
    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # need to pass empty dict (#1329)


    ## 1.a setup BOW
    bow_train   = cv2.BOWKMeansTrainer(10) # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

    basepath = "./Image/BD_simple/"

    for dirname, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if(dirname != basepath):
               try:
                   bow_train.add(feature_sift(os.path.join(dirname, filename),detect, extract))
               except:
                   print('erreur',os.path.join(dirname, filename))
#
    try:
        voc = bow_train.cluster()
        bow_extract.setVocabulary( voc )
    except:
        print('exception vocabulary')

    print(len(bow_extract.getVocabulary()))
    return detect, bow_extract


def bof_model_descriptor(detect,bow_extract):
    traindata, trainlabels = [],[]
    basepath = "./Image/BD_simple/"
    for dirname, dirnames, filenames in os.walk(basepath):
        categorie = dirname.split('/')
        categorie = categorie[len(categorie)-1]
        for filename in filenames:
            if(dirname != basepath):
                try:
                    traindata.extend(feature_bow(os.path.join(dirname,filename),detect,bow_extract))
                    trainlabels.append(int(categorie))
                except:
                    print('erreur',os.path.join(dirname, filename))

    result = {
                  'traindata' : traindata,
                  'trainlabels' : trainlabels
            }
    dataframe = pd.DataFrame(result)
    dataframe.to_csv('descriptor_bof_1.csv')
    dataframe.to_json("descriptor_bof.json")
    return result

def predict_bof(img,train,detect,bow_extract):
    sample = feature_bow2(img,detect,bow_extract)
    clf = neighbors.KNeighborsClassifier(5, weights='distance')
    clf.fit(np.array(train['traindata']), np.array(train['trainlabels']))
    z = clf.predict(sample)
    if z[0] == 1:
        return 'electromenagers'
    elif z[0] == 2:
        return 'materiaux'
    elif z[0] == 3:
        return 'meuble'
    elif z[0] == 4:
        return 'petit electromenagers'
    elif z[0] == 5:
        return 'textile'

def predict_bof1(img,train):
    sample = feature_bow(img)
    clf = neighbors.KNeighborsClassifier(5, weights='distance')
    clf.fit(np.array(train['traindata']), np.array(train['trainlabels']))
    z = clf.predict(sample)
    if z[0] == 1:
        return 'electromenagers'
    elif z[0] == 2:
        return 'materiaux'
    elif z[0] == 3:
        return 'meuble'
    elif z[0] == 4:
        return 'petit electromenagers'
    elif z[0] == 5:
        return 'textile'





def ClassifyImage(Image):
    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # need to pass empty dict (#1329)


    ## 1.a setup BOW
    bow_train   = cv2.BOWKMeansTrainer(10) # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

#    basepath = "./Image/BD_mini/contenant/"
#
#    images = ["1.jpg",
#    "2.jpg",
#    "3.jpg",
#    "4.jpg"]
#
#    """
#    for i in images:
#      bow_train.add(feature_sift(path("left", i)))
#      bow_train.add(feature_sift(path("right",i)))
#    """
#    bow_train.add(feature_sift(os.path.join(basepath, images[0])))
#    bow_train.add(feature_sift(os.path.join(basepath, images[1])))
#    bow_train.add(feature_sift(os.path.join(basepath, images[2])))
#    bow_train.add(feature_sift(os.path.join(basepath, images[3])))

    voc = bow_train.cluster()
    bow_extract.setVocabulary( voc )
