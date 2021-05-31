# import local modules
from dictionary import DONE, SUCCESS, FAILED
# import foreign modules
from zipfile import ZipFile
from os import path, remove, listdir
from requests import get
from cv2 import imread, bitwise_and
import numpy as np

def initilizeDataset():
    print('{:<40s}'.format("Initializing dataset..."), end = "", flush = True)
    if (path.exists("./dataset/leedsbutterfly") == False):
        print(FAILED)
        print('{:<40s}'.format("Requesting compressed data..."), end = "", flush = True)
        url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
        r = get(url, allow_redirects = True)
        open('compressedData.zip', 'wb').write(r.content)
        print(DONE)
        print('{:<40s}'.format("Uncompressing data..."), end = "", flush = True)
        with ZipFile("compressedData.zip", 'r') as file:
            file.extractall("./dataset")
        print(DONE)
        print('{:<40s}'.format("Deleting temporary files..."), end = "", flush = True)
        remove('compressedData.zip')
        print(DONE)
    else:
        print(SUCCESS)

def parseDataset():
    images = []
    masks = []
    labels = []
    print('{:<40s}'.format("Parsing dataset..."), end = "", flush = True)
    for filename in listdir("dataset/leedsbutterfly/images"):
        img = imread(path.join("dataset/leedsbutterfly/images", filename))
        mask = imread(path.join("dataset/leedsbutterfly/segmentations", filename[:-4] + "_seg0.png"), 0)
        if img is not None and mask is not None:
            images.append(img[:,:,::-1])
            masks.append(mask)
            labels.append(int(filename[:3]) - 1)
    print(DONE)
    return images, masks, np.array(labels)

def segmentData(images, masks):
    print('{:<40s}'.format("Segmenting images..."), end = "", flush = True)
    for image in range(len(images)):
        images[image] = bitwise_and(images[image], images[image], mask = masks[image])
    print(DONE)
    return images

def isCached(filename):
    print('{:<40s}'.format("Checking if " + filename + " is cached..."), end = "", flush = True)
    if path.exists(path.join("dataset", filename + '.npy')):
        print(SUCCESS)
        return True
    else:
        print(FAILED)
        return False

def cacheData(data, filename):
    print('{:<40s}'.format("Caching " + filename + "..."), end = "", flush = True)
    np.save(path.join("dataset", filename + '.npy'), data)
    print(DONE)

def loadCachedData(filename):
    print('{:<40s}'.format("Loading cached " + filename + "..."), end = "", flush = True)
    data = np.load(path.join("dataset", filename + '.npy'))
    print(DONE)
    return data

def verifyCachedData(imageSize, filename):
    print('{:<40s}'.format("Verifying cached " + filename + "..."), end = "", flush = True)
    data = np.load(path.join("dataset", filename + '.npy'))
    if data.shape[1] == imageSize and data.shape[2] == imageSize:
        print(SUCCESS)
        return True
    else:
        print(FAILED)
        return False
