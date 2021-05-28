import zipfile
from os import path, remove, listdir
import requests
import cv2
import numpy as np
from skimage import transform

def initilizeDataset():
    print("Initializing dataset...")
    if (path.exists("./dataset/leedsbutterfly") == False):
        print("\tRequesting compressed data, this might take a while...")
        url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
        r = requests.get(url, allow_redirects=True)
        open('compressedData.zip', 'wb').write(r.content)
        print("\tUncompressing data...")
        with zipfile.ZipFile("compressedData.zip", 'r') as zip_ref:
            zip_ref.extractall("./dataset")
        print("\tDeleting temporary files...")
        remove('compressedData.zip')
    print("\tDataset is initializied")

def parseDataset():
    images = []
    masks = []
    labels = []
    print("Parsing dataset...")
    for filename in listdir("dataset/leedsbutterfly/images"):
        img = cv2.imread(path.join("dataset/leedsbutterfly/images", filename))
        mask = cv2.imread(path.join("dataset/leedsbutterfly/segmentations", filename[:-4] + "_seg0.png"), 0)
        if img is not None:
            images.append(img[:,:,::-1])
            masks.append(mask)
            labels.append(int(filename[:3]))
    print("\tDataset is parsed")
    return images, masks, np.array(labels)

def segmentData(images, masks):
    print("Segmenting data...")
    for image in range(len(images)):
        images[image] = cv2.bitwise_and(images[image], images[image], mask = masks[image])
        images[image] = np.array(transform.resize(images[image], (200, 200), mode = "constant"))
    print("\tSegmented " + str(len(images)) + " images")
    return np.array(images)

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
