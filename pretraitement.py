import skimage,os; 
from skimage import io,data,feature,filters;
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi

# l'objectif de cette partie du prétraitement des images est de redimmensionner les images de telle sorte 
# que toutes les images aient la même taille, mais aussi de les convertir en niveaux de gris en vue de 
# simplifier l'algo de classification d'images
#
# Chacune de ces fonctions prend en entrée une image

# Fonction permettant de lire une image
def lireImage(nom_im):
   nom_im=skimage.io.imread(nom_im); 
   return nom_im
    

# Fonction permettant de redimmensionner une image
def redimmensionner(nom_im):
    img=lireImage(nom_im);
    img_=resize(img, (100, 100))
    return img_
   


# Fonction permettant de convertir une image en couleur en niveaux de gris

def conversion_rgb_to_gray(img):
   img=lireImage(img);
   img_gray = rgb2gray(img)
   return img_gray


###########################"#####Test desfonctions créées####################################

#Test de la fonction lireImage
img=lireImage('3480940213994_P.jpg') 
skimage.io.imshow(img)


#Test de la Fonction redimmensionner 
nom_im='3480940213994_P.jpg'
img_=redimmensionner(nom_im)
skimage.io.imshow(img_)
    
#Test de la Fonction conversion_rgb_to_gray 
nom_im='3480940213994_P.jpg'
img_gray=conversion_rgb_to_gray(nom_im)
skimage.io.imshow(img_gray)
###################################### Detection contour image #######################


# Compute the Canny filter for two values of sigma 3 et 5
im=conversion_rgb_to_gray('image2.jpg')
edges1 = feature.canny(im,sigma=3)
edges2 = feature.canny(im, sigma=5)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=3$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=5$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)


# Filtre de Sobel
im=conversion_rgb_to_gray('image2.jpg')
edges = filters.sobel(im)
skimage.io.imshow(edges)


# Nous avons utilisé les filtres de Canny et de Sobel pour détecter les contours de nos images pour ensuite extraire le contour le plus 
# grand; toutefois les résultats obtenus avec ces filtres ne sont pas satisfaisants si l'image est bruitée