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
    GRAY_IMAGE_SIZE,
    GRAY_IMAGE_SPLITS,
    GRAY_IMAGE_TRAIN_SIZE,
    GRAY_IMAGE_EPOCHS,
    GRAY_IMAGE_BATCH_SIZE,
    CROSS_VALIDATION_STATUS
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
import matplotlib.pyplot as plt
from tensorflow import get_logger
from logging import ERROR

# remove warnings outputted by tensorflow during cross validation
get_logger().setLevel(ERROR)


print("\n\nGet dataset:\n\n")


images = None
masks = None
labels = None
data = None

if COLOR_MODEL_STATUS or GRAY_MODEL_STATUS:
    # branch if labels are cached, otherwise get labels
    if isCached(LABEL_FILENAME):
        labels = loadCachedData(LABEL_FILENAME)
    else:
        initilizeDataset()
        # parse dataset to numpy arrays
        images, masks, labels = parseDataset()
        # cache data to reduce loadtime for next runtime
        cacheData(labels, LABEL_FILENAME)
if COLOR_MODEL_STATUS:
    # branch if color images are cached, otherwise get data
    if isCached(COLOR_DATA_FILENAME) and verifyCachedData(COLOR_IMAGE_SIZE, COLOR_DATA_FILENAME):
        dataColor = loadCachedData(COLOR_DATA_FILENAME)
    else:
        if data is None:
            # parse dataset to numpy arrays
            if images is None or masks is None:
                initilizeDataset()
                images, masks, _ = parseDataset()
            # segment images
            data = segmentData(images, masks)
        # resize images
        dataColor = resizeImages(data, COLOR_IMAGE_SIZE)
        # cache dataset to reduce loadtime for next runtime
        cacheData(dataColor, COLOR_DATA_FILENAME)
if GRAY_MODEL_STATUS:
    # branch if gray images are cached, otherwise get data
    if isCached(GRAY_DATA_FILENAME) and verifyCachedData(GRAY_IMAGE_SIZE, GRAY_DATA_FILENAME):
        dataGray = loadCachedData(GRAY_DATA_FILENAME)
    else:
        if data is None:
            # parse dataset to numpy arrays
            if images is None or masks is None:
                initilizeDataset()
                images, masks, _ = parseDataset()
            # segment images
            data = segmentData(images, masks)
        # resize images
        dataGray = resizeImages(data, GRAY_IMAGE_SIZE)
        # convert images to grayscale
        dataGray = grayscaleConverter(dataGray)
        # added axis to handle convolutional layers
        dataGray = np.expand_dims(dataGray, axis = 3)
        # cache dataset to reduce loadtime for next runtime
        cacheData(dataGray, GRAY_DATA_FILENAME)


print("\n\nDefine model:\n\n")


if COLOR_MODEL_STATUS:
    # define and print summary of model
    modelColor = defineModelColor()
    modelColor.summary()
if GRAY_MODEL_STATUS:
    # define and print summary of model
    modelGray = defineModelGray()
    modelGray.summary()


if CROSS_VALIDATION_STATUS:
    print("\n\nValidate model:\n\n")


    if COLOR_MODEL_STATUS:
        # calculate cross validation of model
        calculateCrossValidation(
            defineModelColor, 
            dataColor, 
            labels, 
            COLOR_IMAGE_BATCH_SIZE, 
            COLOR_IMAGE_EPOCHS, 
            COLOR_IMAGE_SPLITS, 
            False
        )
    if GRAY_MODEL_STATUS:
        # calculate cross validation of model
        calculateCrossValidation(
            defineModelGray, 
            dataGray, 
            labels, 
            GRAY_IMAGE_BATCH_SIZE, 
            GRAY_IMAGE_EPOCHS, 
            GRAY_IMAGE_SPLITS, 
            False
        )


print("\n\nTrain model:\n\n")


if COLOR_MODEL_STATUS:
    # prepare data by splitting
    xTrainColor, xTestColor, yTrainColor, yTestColor = prepareData(dataColor, labels, COLOR_IMAGE_TRAIN_SIZE)
    # train model and plot results
    modelColor, results = trainModel(modelColor, xTrainColor, xTestColor, yTrainColor, yTestColor, COLOR_IMAGE_BATCH_SIZE, COLOR_IMAGE_EPOCHS, 0)
    plot(results)
if GRAY_MODEL_STATUS:
    # prepare data by splitting
    xTrainGray, xTestGray, yTrainGray, yTestGray = prepareData(dataGray, labels, GRAY_IMAGE_TRAIN_SIZE)
    # train model and plot results
    modelGray, results = trainModel(modelGray, xTrainGray, xTestGray, yTrainGray, yTestGray, GRAY_IMAGE_BATCH_SIZE, GRAY_IMAGE_EPOCHS, 0)
    plot(results)


print("\n\nPredict model:\n\n")


if COLOR_MODEL_STATUS:
    # output prediction results 
    predictModel(modelColor, xTestColor, yTestColor)
    # generate an image of the model summary
    #generateModelDiagram(modelColor, "modelColor")
if GRAY_MODEL_STATUS:
    # output prediction results 
    predictModel(modelGray, xTestGray, yTestGray)
    # generate an image of the model summary
    #generateModelDiagram(modelGray, "modelGray")
