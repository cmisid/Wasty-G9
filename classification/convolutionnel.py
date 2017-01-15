# Redimmension
import skimage.transform as tr
import os
import numpy as np
import skimage.io as io
import skimage.transform as tr
from random import randrange, uniform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from urllib.request import urlretrieve
import _pickle as pickle
import os
import gzip

import scipy.misc as mi

import numpy as np
import theano
import random
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# 
t=list(range(0,672))
random.shuffle(t)
# uniform gives you a floating-point value
chemin="/home/etudiant/BD/"
id_test = []

X_test = np.zeros((672-570 ,100, 100, 3))
Y_test = np.zeros((672-570))

X_train = np.zeros((570, 100, 100, 3))
Y_train = np.zeros((570)) 


i = 0

test_index = 0
train_index = 0
# Catég
index=0
i=0
for categorie in os.listdir(chemin):
    if not os.path.isfile(categorie):
        file_path = os.path.join(chemin, categorie)
        for image in os.listdir(file_path):
            ID_image = int(image[:-4])
            if i in t[:570]:
                X_train[train_index] = mi.imresize(io.imread(os.path.join(file_path\
                , image)), (100, 100,3))
                Y_train[train_index]=index
                train_index+=1
            else:
                X_test[test_index] = mi.imresize(io.imread(os.path.join(file_path\
                , image)), (100, 100,3))
                Y_test[test_index]=index
                test_index+=1
                
            i+=1
         
    index+=1

Y_train = Y_train.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)
plt.imshow(X_train[1])
plt.show()


net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 100, 100,3),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1_pool_size
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=128,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=5 ,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )

# Train the network
nn = net1.fit(X_train, Y_train)
#preds = net1.predict(X_test)
#cm = confusion_matrix(y_test, preds)


