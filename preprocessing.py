import numpy as np
import cv2

def grayscaleConverter(images):
    grayscale = np.zeros(images.shape[:-1])
    for i in range(images.shape[0]): 
        grayscale[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
    return grayscale
