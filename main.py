# import local modules
from dictionary import (
    LABEL_FILENAME,
    COLOR_DATA_FILENAME,
    COLOR_IMAGE_FILENAME,
    COLOR_IMAGE_SIZE, 
    COLOR_IMAGE_EPOCHS, 
    COLOR_IMAGE_BATCH_SIZE, 
    COLOR_IMAGE_SPLITS, 
    COLOR_IMAGE_TRAIN_SIZE,
    GRAY_DATA_FILENAME,
    GRAY_IMAGE_FILENAME,
    GRAY_IMAGE_SIZE,
    GRAY_IMAGE_SPLITS,
    GRAY_IMAGE_TRAIN_SIZE,
    GRAY_IMAGE_EPOCHS,
    GRAY_IMAGE_BATCH_SIZE,
    CROSS_VALIDATION_STATUS,
    CONVERT_TO_GRAYSCALE
)
from datasetParser import (
    initilizeDataset, 
    parseDataset, 
    segmentData, 
    isCached, 
    cacheData, 
    loadCachedData, 
    verifyCachedData
)
from model import (
    prepareData, 
    plot, 
    defineModelColor, 
    defineModelGray, 
    calculateCrossValidation, 
    trainModel, 
    predictModel, 
    generateModelDiagram
)
from preprocessing import grayscaleConverter, resizeImages
# import foreign modules
import numpy as np
from tensorflow import get_logger
from logging import ERROR

# remove warnings outputted by tensorflow during cross validation
get_logger().setLevel(ERROR)

# set data according to model type
if CONVERT_TO_GRAYSCALE:
    DATA_FILENAME = GRAY_DATA_FILENAME
    IMAGE_FILENAME = GRAY_IMAGE_FILENAME
    IMAGE_SIZE = GRAY_IMAGE_SIZE
    BATCH_SIZE = GRAY_IMAGE_BATCH_SIZE
    EPOCHS = GRAY_IMAGE_EPOCHS
    SPLITS = GRAY_IMAGE_SPLITS
    TRAIN_SIZE = GRAY_IMAGE_TRAIN_SIZE
    DEFINE_MODEL = defineModelGray
else:
    DATA_FILENAME = COLOR_DATA_FILENAME
    IMAGE_FILENAME = COLOR_IMAGE_FILENAME
    IMAGE_SIZE = COLOR_IMAGE_SIZE
    BATCH_SIZE = COLOR_IMAGE_BATCH_SIZE
    EPOCHS = COLOR_IMAGE_EPOCHS
    SPLITS = COLOR_IMAGE_SPLITS
    TRAIN_SIZE = COLOR_IMAGE_TRAIN_SIZE
    DEFINE_MODEL = defineModelColor


images = None
masks = None
labels = None
data = None
# branch if labels are cached, otherwise get labels
if isCached(LABEL_FILENAME):
    labels = loadCachedData(LABEL_FILENAME)
else:
    initilizeDataset()
    # parse dataset to numpy arrays
    images, masks, labels = parseDataset()
    # cache data to reduce loadtime for next runtime
    cacheData(labels, LABEL_FILENAME)
# branch if data is cached, otherwise get data
if isCached(DATA_FILENAME) and verifyCachedData(IMAGE_SIZE, DATA_FILENAME):
    data = loadCachedData(DATA_FILENAME)
else:
    # parse dataset to numpy arrays
    if images is None or masks is None:
        initilizeDataset()
        images, masks, _ = parseDataset()
    # segment images
    data = segmentData(images, masks)
    # resize images
    data = resizeImages(data, IMAGE_SIZE)
    if CONVERT_TO_GRAYSCALE:
        # convert images to grayscale
        data = grayscaleConverter(data)
        # added axis to handle convolutional layers
        data = np.expand_dims(data, axis = 3)
    # cache dataset to reduce loadtime for next runtime
    cacheData(data, DATA_FILENAME)
print('\n')


# define and print summary of model
model = DEFINE_MODEL()
model.summary()
# generate an image of the model summary
#generateModelDiagram(model, IMAGE_FILENAME)
print('\n')


# checks if cross validation variable is set to true
if CROSS_VALIDATION_STATUS:
    # calculate cross validation of model
    calculateCrossValidation(
        DEFINE_MODEL, 
        data, 
        labels, 
        BATCH_SIZE, 
        EPOCHS, 
        SPLITS, 
        False
    )


# prepare data by splitting
xTrain, xTest, yTrain, yTest = prepareData(data, labels, TRAIN_SIZE)
# train model and plot results
print("\n\nTraining model...")
model, results = trainModel(model, xTrain, xTest, yTrain, yTest, BATCH_SIZE, EPOCHS, 0)
plot(results)
print('\n')


# output prediction results 
predictModel(model, xTest, yTest)
