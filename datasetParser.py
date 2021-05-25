import zipfile
from os import path, remove
import requests

if (path.exists("./dataset") == False):
    print("Didn't find dataset...")
    print("Requesting compressed data, this might take a while...")
    url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
    r = requests.get(url, allow_redirects=True)
    open('compressedData.zip', 'wb').write(r.content)
    print("Uncompresses data...")
    with zipfile.ZipFile("compressedData.zip", 'r') as zip_ref:
        zip_ref.extractall("./dataset")
    print("Removing compressed data...")
    remove('compressedData.zip')
    print("Dataset is initializied!")
else:
    print("Found dataset!")
