# import foreign modules
from zipfile import ZipFile
from os import path, remove, listdir
from requests import get
from cv2 import imread, bitwise_and
import numpy as np

def initilizeDataset():
    print("Initializing dataset...")
    if (path.exists("./dataset/leedsbutterfly") == False):
        print("\tRequesting compressed data, this might take a while...")
        url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
        r = get(url, allow_redirects = True)
        open('compressedData.zip', 'wb').write(r.content)
        print("\tUncompressing data...")
        with ZipFile("compressedData.zip", 'r') as file:
            file.extractall("./dataset")
        print("\tDeleting temporary files...")
        remove('compressedData.zip')
    print("\tDataset is initializied")

def parseDataset():
    images = []
    masks = []
    labels = []
    print("Parsing dataset...")
    for filename in listdir("dataset/leedsbutterfly/images"):
        img = imread(path.join("dataset/leedsbutterfly/images", filename))
        mask = imread(path.join("dataset/leedsbutterfly/segmentations", filename[:-4] + "_seg0.png"), 0)
        if img is not None and mask is not None:
            images.append(img[:,:,::-1])
            masks.append(mask)
            labels.append(int(filename[:3]) - 1)
    print("\tSuccessfully parsed dataset")
    return images, masks, np.array(labels)

def segmentData(images, masks):
    print("Segmenting images...")
    for image in range(len(images)):
        images[image] = bitwise_and(images[image], images[image], mask = masks[image])
    print("\tSuccessfully segmented " + str(len(images)) + " images")
    return images

def isCached(filename):
    print("Checking if " + filename + " is cached...")
    if path.exists(path.join("dataset", filename + '.npy')):
        print("\tSuccessfully found cached " + filename)
        return True
    else:
        print("\tFailed to find cached " + filename)
        return False

def cacheData(data, filename):
    print("Caching " + filename + "...")
    np.save(path.join("dataset", filename + '.npy'), data)
    print("\tSuccessfully cached" + filename)

def loadCachedData(filename):
    print("Loading cached " + filename + "...")
    data = np.load(path.join("dataset", filename + '.npy'))
    print("\tSuccessfully loaded cached " + filename)
    return data

def verifyCachedData(imageSize, filename):
    print("Verifying cached " + filename + "...")
    data = np.load(path.join("dataset", filename + '.npy'))
    if data.shape[1] == imageSize and data.shape[2] == imageSize:
        print("\tSuccessfully verified cached " + filename)
        return True
    else:
        print("\tFailed to verify cached " + filename)
        return False
