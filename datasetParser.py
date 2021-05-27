import zipfile
from os import path, remove, listdir
import requests
import cv2

def initilizeDataset():
    print("initializing dataset...")
    if (path.exists("./dataset") == False):
        print("Requesting compressed data, this might take a while...")
        url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
        r = requests.get(url, allow_redirects=True)
        open('compressedData.zip', 'wb').write(r.content)
        print("Uncompressing data...")
        with zipfile.ZipFile("compressedData.zip", 'r') as zip_ref:
            zip_ref.extractall("./dataset")
        print("Deleting temporary files...")
        remove('compressedData.zip')
    print("Dataset is initializied")

def parseDataset():
    images = []
    segmentations = []
    labels = []
    print("Parsing dataset...")
    for filename in listdir("dataset/leedsbutterfly/images"):
        img = cv2.imread(path.join("dataset/leedsbutterfly/images", filename))
        seg = cv2.imread(path.join("dataset/leedsbutterfly/segmentations", filename[:-4] + "_seg0.png"))
        if img is not None:
            images.append(img)
            segmentations.append(seg)
            labels.append(int(filename[:3]))
    print("Dataset is parsed")
    return images, segmentations, labels
