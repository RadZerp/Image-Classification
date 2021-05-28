from datasetParser import initilizeDataset, parseDataset, segmentData, datasetIsCached, cacheDataset, loadCachedDataset
from matplotlib import pyplot as plt

if datasetIsCached():
    data, labels = loadCachedDataset()
else:
    initilizeDataset()
    images, masks, labels = parseDataset()
    data = images
    data = segmentData(images, masks)
    cacheDataset(data, labels)
plt.imshow(data[0])

from model import prepareData, defineModel
#data = numpy.array(data)
X_train, X_test, y_train, y_test = prepareData(data, labels)
model = defineModel()
results = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 1, epochs = 10, verbose = 1)
