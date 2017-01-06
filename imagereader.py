#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:20:43 2017

@author: ismailaddou
"""

#import argparse
import cv2
from skimage import data, color, util
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import hog
#
#parser = argparse.ArgumentParser()
#parser.add_argument("path", type=int, help="le chemin de l'image") 
#img = cv2.imread(args.path)
#cv2.imshow("title", img)


img1 = cv2.imread('BD_p/contenant/1.jpg',0)           

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)


#HOG 

def task(image):
    """
    Apply some functions and return an image.
    """
    image = denoise_tv_chambolle(image[0][0], weight=0.1, multichannel=True)
    fd, hog_image = hog(color.rgb2gray(image), orientations=8,
                        pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                        visualise=True)
    return hog_image


# Prepare images
hubble = data.hubble_deep_field()
width = 10
pics = util.view_as_windows(hubble, (width, hubble.shape[1], hubble.shape[2]), step=width)



# Initiate SIFT detector 