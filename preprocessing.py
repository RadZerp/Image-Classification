# import local modules
from dictionary import IMAGE_SIZE
# import foreign modules
from numpy import zeros
from cv2 import cvtColor, COLOR_RGB2GRAY
import numpy as np
from skimage import transform

def resizeImages(images):
    print("Resizing images...")
    for image in range(len(images)):
        images[image] = np.array(transform.resize(images[image], (IMAGE_SIZE, IMAGE_SIZE), mode = "constant"))
    print("\tSuccessfully resized images")
    return np.array(images)

def grayscaleConverter(images):
    grayscale = zeros(images.shape[:-1])
    for i in range(images.shape[0]): 
        grayscale[i] = cvtColor(images[i].astype('float32'), COLOR_RGB2GRAY)
    return grayscale
