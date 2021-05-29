# import local modules
from datasetParser import initilizeDataset, parseDataset, segmentData, datasetIsCached, cacheDataset, loadCachedDataset, verifyCachedDataset
from model import prepareData, plot, defineModel, defineGrayscaleModel, calculateCrossValidation, trainModel, predictModel, generateModelDiagram
from preprocessing import grayscaleConverter
# import foreign modules
from tensorflow import get_logger
from logging import ERROR

# remove warnings outputted by tensorflow during cross validation
get_logger().setLevel(ERROR)


print("\n\nGet dataset:\n\n")


# branch if dataset is cached, otherwise initilize dataset
if datasetIsCached() and verifyCachedDataset():
    data, labels = loadCachedDataset()
else:
    initilizeDataset()
    # parse dataset to numpy arrays
    images, masks, labels = parseDataset()
    # segment images
    data = segmentData(images, masks)
    # cache dataset to reduce loadtime for next runtime
    cacheDataset(data, labels)
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
calculateCrossValidation(defineModel, data, labels, 0)
# calculate cross validation on grayscale model
#calculateCrossValidation(defineGrayscaleModel, grayscaleData, labels, 0)


print("\n\nTrain model:\n\n")


# prepare data by splitting
xTrain, xTest, yTrain, yTest = prepareData(data, labels)
# train model and plot results
model, results = trainModel(model, xTrain, xTest, yTrain, yTest, 0)
plot(results)


print("\n\nValidate model:\n\n")


# output prediction results
predictModel(model, xTest, yTest)
# generate an image of the model summary
#generateModelDiagram(model)
