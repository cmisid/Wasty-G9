import os
import cv2
import json


sift = cv2.xfeatures2d.SIFT_create()
descripteurs = []
images = []
i = 0
for dirname, dirnames, filenames in os.walk("/Users/ismailaddou/Wasty-ImageClassifier/BD_image_pro"):
    for filename in filenames:
        i = i + 1
        path = os.path.join(dirname,filename) 
        img1 = cv2.imread(path,0) 
        kp, des = sift.detectAndCompute(img1,None)
        image = dict()
        image = {
                     'indice' : i,
                     'category' : dirname,
                     "desc_sift" : des1.tolist()
                 }
        images.append(image)

#print(descripteurs)

output = json.dumps(images)
output = json.loads(output)
with open('descriptors.json', 'w') as outfile:
    json.dump(output, outfile)
