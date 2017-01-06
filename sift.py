import os

for root, dirs, files in os.walk('/Users/ismailaddou/Wasty-ImageClassifier/Wasty-ImageClassifier/BD_image_pro/conteneur'):
    print(root)
    print(dirs)
    print(files)
    for name in files:
        print(name)
 
for dirname, dirnames, filenames in os.walk('/Users/ismailaddou/Documents/Wasty-ImageClassifier/BD_image_pro/conteneur'):

    for filename in filenames:
        print(os.path.join(dirname,filename))