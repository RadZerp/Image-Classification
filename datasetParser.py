import zipfile
from os import path, remove
import requests

def initilizeDataset():
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
