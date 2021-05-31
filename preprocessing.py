# import local modules
from dictionary import DONE
# import foreign modules
from numpy import zeros
from cv2 import cvtColor, COLOR_RGB2GRAY
import numpy as np
from skimage import transform

# resizes images to get uniform size
def resizeImages(images, imageSize):
    print('{:<40s}'.format("Resizing images to " + str(imageSize) + "x" + str(imageSize) + "..."), end = "", flush = True)
    for i in range(len(images)):
        images[i] = np.array(transform.resize(images[i], (imageSize, imageSize), mode = "constant"))
    print(DONE)
    return np.array(images)

# converts images to grayscale
def grayscaleConverter(images):
    print('{:<40s}'.format("Converting images to grayscale..."), end = "", flush = True)
    grayscale = zeros(images.shape[:-1])
    for i in range(images.shape[0]): 
        grayscale[i] = cvtColor(images[i].astype('float32'), COLOR_RGB2GRAY)
    print(DONE)
    return grayscale
