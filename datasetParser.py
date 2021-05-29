# import local modules
from dictionary import IMAGE_SIZE
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
    print("\tDataset is parsed")
    return images, masks, np.array(labels)

def segmentData(images, masks):
    print("Segmenting data...")
    for image in range(len(images)):
        images[image] = bitwise_and(images[image], images[image], mask = masks[image])
    print("\tSegmented " + str(len(images)) + " images")
    return images

def datasetIsCached():
    print("Checking if dataset is cached...")
    if path.exists("./dataset/data.npy") and path.exists('./dataset/labels.npy'):
        print("\tFound cached dataset")
        return True
    else:
        print("\tDidn't find cached dataset")
        return False

def cacheDataset(data, labels):
    print("Caching dataset...")
    np.save('./dataset/data.npy', data)
    np.save('./dataset/labels.npy', labels)
    print("\tDataset is cached")

def loadCachedDataset():
    print("Loading cached dataset...")
    data = np.load('./dataset/data.npy')
    labels = np.load('./dataset/labels.npy')
    print("\tDataset is loaded")
    return data, labels

def verifyCachedDataset():
    print("Verifying cached dataset...")
    data = np.load('./dataset/data.npy')
    if data.shape[1] == IMAGE_SIZE and data.shape[2] == IMAGE_SIZE:
        print("\tSuccessfully verified cached dataset")
        return True
    else:
        print("\tFailed to verify cached dataset")
        return False
