# import foreign modules
from numpy import zeros
from cv2 import cvtColor, COLOR_RGB2GRAY

def grayscaleConverter(images):
    grayscale = zeros(images.shape[:-1])
    for i in range(images.shape[0]): 
        grayscale[i] = cvtColor(images[i].astype('float32'), COLOR_RGB2GRAY)
    return grayscale
