# import local modules
from dictionary import (
    LABEL_FILENAME,
    COLOR_MODEL_STATUS,
    COLOR_DATA_FILENAME,
    COLOR_IMAGE_SIZE, 
    COLOR_IMAGE_EPOCHS, 
    COLOR_IMAGE_BATCH_SIZE, 
    COLOR_IMAGE_SPLITS, 
    COLOR_IMAGE_TRAIN_SIZE,
    GRAY_MODEL_STATUS,
    GRAY_DATA_FILENAME,
    GRAY_IMAGE_SIZE
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
    defineModel, 
    defineGrayscaleModel, 
    calculateCrossValidation, 
    trainModel, 
    predictModel, 
    generateModelDiagram
)
from preprocessing import grayscaleConverter, resizeImages
# import foreign modules
from tensorflow import get_logger
from logging import ERROR

# remove warnings outputted by tensorflow during cross validation
get_logger().setLevel(ERROR)


print("\n\nGet dataset:\n\n")


# branch if dataset is cached, otherwise initilize dataset
if COLOR_MODEL_STATUS and isCached(COLOR_DATA_FILENAME) and verifyCachedData(COLOR_IMAGE_SIZE, COLOR_DATA_FILENAME):
    data = loadCachedData(COLOR_DATA_FILENAME)
    labels = loadCachedData(LABEL_FILENAME)
else:
    initilizeDataset()
    # parse dataset to numpy arrays
    images, masks, labels = parseDataset()
    # segment images
    images = segmentData(images, masks)
    # resize images
    data = resizeImages(images, COLOR_IMAGE_SIZE)
    # cache dataset to reduce loadtime for next runtime
    cacheData(data, COLOR_DATA_FILENAME)
    cacheData(labels, LABEL_FILENAME)
# convert images to grayscale
grayscaleData = grayscaleConverter(data)


print("\n\nDefine model:\n\n")


# define and print summary of model
model = defineModel()
model.summary()
# define and print summary of grayscale model
#grayscaleModel = defineGrayscaleModel()
#grayscaleModel.summary()


print("\n\nValidate dataset:\n\n")


# calculate cross validation on model
calculateCrossValidation(defineModel, data, labels, COLOR_IMAGE_BATCH_SIZE, COLOR_IMAGE_EPOCHS, COLOR_IMAGE_SPLITS, 0)
# calculate cross validation on grayscale model
#calculateCrossValidation(defineGrayscaleModel, grayscaleData, labels, 0)


print("\n\nTrain model:\n\n")


# prepare data by splitting
xTrain, xTest, yTrain, yTest = prepareData(data, labels, COLOR_IMAGE_TRAIN_SIZE)
# train model and plot results
model, results = trainModel(model, xTrain, xTest, yTrain, yTest, COLOR_IMAGE_BATCH_SIZE, COLOR_IMAGE_EPOCHS, 0)
plot(results)


print("\n\nValidate model:\n\n")


# output prediction results
predictModel(model, xTest, yTest)
# generate an image of the model summary
#generateModelDiagram(model)
