# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:40:45 2019

@author: MitTal
"""

import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from l0smooth.L0_serial import l0_smooth
import src


image = cv2.imread('images\\img.png')
gray_img = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
image_l0 = l0_smooth(image)
image_blur = cv2.GaussianBlur(image_l0, (5, 5), 0)
img_can = cv2.Canny(image_blur.astype('uint8'), 100, 200)

gray_blur = cv2.cvtColor(image_blur,cv2.COLOR_BGR2GRAY)

khali_image = np.zeros((232,230))
for i in range(0,len(khali_image)):
    for j in range(0,len(khali_image[0])):
        if(img_can[i][j]==255):
            khali_image[i][j] = gray_blur[i][j]

image_I2 = np.zeros((232,230))

for i in range(0,len(image_I2)):
    for j in range(0,len(image_I2[0])):

        if(gray_blur[i][j] == 0):
            delta_I0 = 255 
        else:
            delta_I0 = ((255-gray_blur[i][j])/gray_blur[i][j])*gray_img[i][j]

        image_I2[i][j] = min(255, (gray_img[i][j] + delta_I0))

cv2.imwrite('abc.png',image_I2)



image_I2 = image_I2.astype('uint8')

ret2,th2 = cv2.threshold(image_I2.flatten(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#img_thresh_Gaussian = cv2.adaptiveThreshold(image_I2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

threshold_img = np.zeros((232,230))
for i in range(0,len(image_I2)):
    for j in range(0,len(image_I2[0])):
        if(image_I2[i][j] >= ret2):
            threshold_img[i][j] = 255
        else:
            threshold_img[i][j] = 0
            
cv2.imwrite('threshold_image.png',threshold_img)
