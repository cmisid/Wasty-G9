
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


seed = 7
numpy.random.seed(seed)
#Lecture du dataframe contenant 49 decripteurs SIFT de chaque image 
dataframe = pandas.read_csv("/home/etudiant/model_8000im_4categories_k100.csv", header=None)
dataset = dataframe.values    


#Division des des données descripteurs(X) et la colonne des labels(Y)
X = dataset[:,1:50]
Y = dataset[:,51]


#Conversion de labels en entiers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#Conversion des labels en variables catégorielles
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    	# Création du réseau de neuronnes
	model = Sequential()
     # Création de 49 neuronnes dans la couche d'entrée et de 6 couches cachées
	model.add(Dense(6, input_dim=49, init='normal', activation='relu'))
     # Création de 5 neuronnes dans la couche de sortie
	model.add(Dense(5, init='normal', activation='sigmoid'))
   	# Compilation du modéle
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#Lancement du classifieur avec 200 itérations et 100 observations prises à chaque rétro propagation
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=100, verbose=0)
#Validation croisée avec 10-fold pour évaluaer le modéle
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


#Affichage des résultats
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))