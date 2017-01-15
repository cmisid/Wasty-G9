#Groupe [9]

import os
import cv2
import json
import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

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
    for dirname, dirnames, filenames in os.walk("./Image/BD_mini1/"):
        for filename in filenames:
            if(dirname != './Image/BD_mini1/'):
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
                    print(category)
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

        # Calculer le nombre de correspondance et filtrer par précision
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
#Entrée : sorted results de la fonction predict_class
#Description : Ordonne par moyenne de nombre total de correspondance par classe
#           la liste des catégories
#Sortie : Une liste du plus probable au moin probable
def compute(sorted_results):
    df = pd.DataFrame(sorted_results)
    means = df.groupby('category').mean()
    category = means[means['nb_matches'] == means.nb_matches.max()].index.tolist()
    sorted_list = means.sort_index(by=['nb_matches'],ascending=[False]).index.tolist()
    return sorted_list

def path(cls,i): # "./left03.jpg"
  return "%s/%s%02d.jpg"  % (datapath,cls,i+1)

#Entrée : chemin de l'image
#Description : Extraction des features de l'image
#Sortie : np array de features
def feature_sift(fn,detect,extract):
  im = cv2.imread(fn,0)
  im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
  return extract.compute(im, detect.detect(im))[1]

#Entrée : chemin de l'image, detect et extract retournés par bof_train_extract_features()
#Description : Extraction des features de l'image
#Sortie : np array de features
def feature_bow(fn,detect,bow_extract):

  im = cv2.imread(fn,0)
  im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
  return bow_extract.compute(im, detect.detect(im))

#Entrée : Image en np-array en couleurs, detect et extract retournés par bof_train_extract_features()
#Description : Extraction des features de l'image
#Sortie : np array de features
def feature_bow_npimg(im,detect,bow_extract):
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
    return bow_extract.compute(image_gray, detect.detect(image_gray))


#Description : deuxième méthode de classification, bag of features
#Sortie : objets réutilisable pour stocker l'apprentissage
def bof_train_extract_features():
    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    # Flann méthode de match entre les keypoints pour former des mots visuels
    flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # need to pass empty dict (#1329)


    ## Préparation du bag of words
    bow_train   = cv2.BOWKMeansTrainer(50) # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

    basepath = "./Image/BD_simple/"

    # Entrainer toutes les images
    for dirname, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if(dirname != basepath):
               try:
                   bow_train.add(feature_sift(os.path.join(dirname, filename),detect, extract))
               except:
                   print('skipped image [ bof_train_extract_features() ]',os.path.join(dirname, filename))
    #
#    test = bow_train.getDescriptors()
#
#    features = []
#    for item in test :
#        t = item.tolist()
#        features.extend(t)
#
#    features = np.array(features,dtype=np.float32)
#    try:
#        clusterer = KMeans(n_clusters=18, random_state=10)
#        cluster_labels = clusterer.fit_predict(features)
#        voc = np.array(clusterer.cluster_centers_, dtype=np.float32)
#        bow_extract.setVocabulary(voc)
#    except:
#        print('exception vocab')
    try:
        voc = bow_train.cluster()
        bow_extract.setVocabulary( voc )
    except:
        print('exception vocabulary')

    return detect, bow_extract,voc, bow_train

#Description : deuxième méthode de classification, bag of features
#Sortie : dictionnaire qui contient les données d'entrainnement et les classes associées
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
#                    trainlabels.extend(categorie)
                except:
                    print('skipped image [ bof_model_descriptor(detect,bow_extract) ]',os.path.join(dirname, filename))

    result = {
                  'traindata' : traindata,
                  'trainlabels' : trainlabels
            }
#    dataframe = pd.DataFrame(result)
#    dataframe.to_csv('descriptor_bof_1.csv')
#    dataframe.to_json("descriptor_bof.json")
    return result

#Entree: img en nparray, detect et bow_extract sont retournées par la fonction
#   bof_train_extract_features()
#Description : Fonction pour prédire, utilisable directement après l'entrainement
#Sortie : nom de la classe
def predict_bof(img,train,detect,bow_extract):
    print('Predicting ...')
    sample = feature_bow_npimg(img,detect,bow_extract)

    #Méthodes des classifications
    clf = AdaBoostClassifier()
#    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#    clf = neighbors.KNeighborsClassifier(5, weights='distance')
#    clf = DecisionTreeClassifier(random_state=0)

    #prédiction
    clf.fit(np.array(train['traindata']), np.array(train['trainlabels']))

    z = clf.predict(sample)
#    if z[0] == 1:
#        return 'deco'
#    elif z[0] == 2:
#        return 'divers'
#    elif z[0] == 3:
#        return 'electromenager'
#    elif z[0] == 4:
#        return 'jardin'
#    elif z[0] == 5:
#        return 'mobilier'
#    elif z[0] == 6:
#        return 'petit_electromenager'
#    elif z[0] == 7:
#        return 'textile'
#    elif z[0] == 8:
#        return 'transport'
#    elif z[0] == 9:
#        return 'vaisselle'
    #z = clf.predict_proba(sample)
#    return sample
########
    if z[0] == 1:
       return 'electromenagers'
    elif z[0] == 2:
       return 'meuble'
    elif z[0] == 3:
       return 'petit electromenagers'
    elif z[0] == 4:
       return 'textile'
########
    # if z[0] == 1:
    #     return 'four'
    # elif z[0] == 2:
    #     return 'armoire'
    # elif z[0] == 3:
    #     return 'aspirateur'
    # elif z[0] == 4:
    #     return 'bureau'
    # elif z[0] == 5:
    #     return 'frigo'
    # elif z[0] == 6:
    #     return 'chaussures'
    # elif z[0] == 7:
    #     return 'sac'
    # elif z[0] == 9:
    #     return 'lave linge'

#Entree: chemin local de l'image, detect et bow_extract sont retournées par la fonction
#   bof_train_extract_features()
#Description : Fonction de test pour prédire, utilisable directement après l'entrainement
#Sortie : nom de la classe
def predict_bof1(img,train):
    sample = feature_bow(img)
    #clf = neighbors.KNeighborsClassifier(5, weights='distance')
    #clf = DecisionTreeClassifier(random_state=0)
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

def bow_train_model():

    basepath = "./Image/BD_objet1/"

    training_names = os.listdir(basepath)

    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    categories = []
    index = 1
    for training_name in training_names:
        dir = os.path.join(basepath, training_name)
        dir = dir.split('/')
        categorie = dir[len(dir)-1]
        if(categorie != '.DS_Store'):
            categories.append((index,categorie))
            index += 1
    #    class_path = imutils.imlist(dir)
    #    image_paths+=class_path
    #    image_classes+=[class_id]*len(class_path)
    #    class_id+=1

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.xfeatures2d.SIFT_create()
    des_ext = cv2.xfeatures2d.SIFT_create()
    # List where all the descriptors are stored
    des_list = []

    for dirname, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if(dirname != basepath):
                im = cv2.imread(os.path.join(dirname, filename))
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)
                kpts = fea_det.detect(im)
                kpts, des = des_ext.compute(im, kpts)
                des_list.append((os.path.join(dirname, filename), des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    image_classes = []
    count = 0
    for image_path, descriptor in des_list[1:]:
            #    print(descriptor.shape)
        try:

            descriptors = np.vstack((descriptors, descriptor))

            categorie = image_path.split('/')
            categorie = categorie[len(categorie)-2]
            if(categorie == 'sac'):
                image_classes.append(7)
            elif(categorie == 'armoire'):
                image_classes.append(2)
            elif(categorie == 'aspirateur'):
                image_classes.append(3)
            elif(categorie == 'bureau'):
                image_classes.append(4)
            elif(categorie == 'chaussure'):
                image_classes.append(6)
            elif(categorie == 'four'):
                image_classes.append(1)
            elif(categorie == 'frigo'):
                image_classes.append(5)
            elif(categorie == 'lave_linge'):
                image_classes.append(8)
            count += 1
        except:
            print('error skip',image_path)

    # Perform k-means clustering
    k = 100
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((count, k), "float32")
    #for i in xrange(len(image_paths)):
    for i in range(count):
        try:
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1
        except:
            print('error skip',i)


    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return im_features,voc,stdSlr,k

def predict_bow(img,voc,stdSlr,k):
    #basepath = "./Image/BD_test/four.png"

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.xfeatures2d.SIFT_create()
    des_ext = cv2.xfeatures2d.SIFT_create()
    # List where all the descriptors are stored
    des_list = []

    # im = cv2.imread(os.path.join(dirname, filename))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA)

    kpts = fea_det.detect(img)
    kpts, des = des_ext.compute(img, kpts)
    des_list.append(('imgtest', des))

    count = 1
    k = 100
    # Calculate the histogram of features
    img_features = np.zeros((count, k), "float32")
    #for i in xrange(len(image_paths)):

    words, distance = vq(des_list[0][1],voc)
    for w in words:
        img_features[0][w] += 1


    # Scaling the words
    img_features = stdSlr.transform(img_features)
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))
    z = clf.predict(img_features)
    if(z[0] == 7):
        return 'sac'
    elif(z[0] == 2):
        return 'armoire'
    elif(z[0] == 3):
        return 'aspirateur'
    elif(z[0] == 4):
        return 'bureau'
    elif(z[0] == 6):
        return 'chaussure'
    elif(z[0] == 1):
        return 'four'
    elif(z[0] == 5):
        return 'frigo'
    elif(z[0] == 8):
        return 'lave linge'
    return z
