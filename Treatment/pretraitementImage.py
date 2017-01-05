import skimage,os; 
from skimage import io,data;
from skimage.transform import resize
from skimage.color import rgb2gray


# l'objectif de cette partie du prétraitement des images est de redimmensionner les images de telle sorte 
# que toutes les images aient la même taille, mais aussi de les convertir en niveaux de gris en vue de 
# simplifier l'algo de classification d'images
#
# Chacune de ces fonctions prend en entrée une image
# Fonction permettant de redimmensionner une image
def redimmensionner(nom_im):
    img=skimage.io.imread(nom_im);
    img_=resize(img, (100, 100))
    return img_
   


# Fonction permettant de convertir une image en couleur en niveaux de gris

def conversion_rgb_to_gray(img):
    img=skimage.io.imread(img)
    img_gray = rgb2gray(img)
    return img_gray


#Test desfonctions créées; chaque fonction est testée pour voir si les résulats attendus sont 
# conformes à ceux obtenus 
#Test de la Fonction redimmensionner 
nom_im='3480940213994_P.jpg'
img_=redimmensionner(nom_im)
skimage.io.imshow(img_)
    
#Test de la Fonction conversion_rgb_to_gray 
nom_im='3480940213994_P.jpg'
img_gray=conversion_rgb_to_gray(nom_im)
skimage.io.imshow(img_gray)


# Test des deux fonctions: l'image convertie en niveau de gris sera redimensionné
img='3480940213994_P.jpg'
img_gray=conversion_rgb_to_gray(img)
img_gray=skimage.io.imsave('img_gray.jpg',img_gray)
img_g='img_gray.jpg'
img_gray_redim=redimmensionner(img_g)
skimage.io.imshow(img_gray_redim)
img_gray_redim

