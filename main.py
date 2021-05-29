from datasetParser import initilizeDataset, parseDataset, segmentData, datasetIsCached, cacheDataset, loadCachedDataset, verifyCachedDataset
from model import prepareData, plot
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from tensorflow import keras
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.core import *
import logging
#remove warnings outputted by tensorflow during cross validation
tf.get_logger().setLevel(logging.ERROR)
#set parameters
imageSize = 50
trainSize = 0.80
iterations = 50
batchSize = 512
labelNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

print("\n\nGet dataset:\n\n")

if datasetIsCached() and verifyCachedDataset(imageSize):
    data, labels = loadCachedDataset()
else:
    initilizeDataset()
    images, masks, labels = parseDataset()
    data = images
    data = segmentData(images, masks, imageSize)
    cacheDataset(data, labels)

print("\n\nDefine model:\n\n")

def defineModel():
    model = Sequential([
        #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        
        Conv2D(32, (3, 3), input_shape = (imageSize, imageSize, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Conv2D(32, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Flatten(),

        Dense(64, activation = 'relu'),
        Dropout(0.5),

        Dense(10, activation = 'sigmoid')
    ])
    opt = keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = opt, 
        metrics = ['accuracy']
    )
    return model

model = defineModel()
model.summary()

print("\n\nValidate dataset:\n\n")

print("Running cross validation...")
crossVal_model = tf.keras.wrappers.scikit_learn.KerasClassifier(
    build_fn = defineModel, 
    epochs = iterations, 
    batch_size = batchSize, 
    verbose = 0
)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0) 
results = cross_val_score(crossVal_model, data, labels, cv = kfold)
print("Cross validation results: " + str(results))
print("%0.2f accuracy with a standard deviation of %0.2f" % (results.mean(), results.std()))

print("\n\nTrain model:\n\n")

xTrain, xTest, yTrain, yTest = prepareData(data, labels, trainSize)
results = model.fit(
    xTrain, 
    yTrain, 
    validation_data = (xTest, yTest), 
    batch_size = batchSize, 
    epochs = iterations,
    verbose = 0
)
model.evaluate(xTest, yTest)
plot(results)

print("\n\nValidate model:\n\n")

yPred = model.predict(xTest)
yPred = np.argmax(yPred, axis = 1)
yTest = np.argmax(yTest, axis = 1)
print(classification_report(yTest, yPred, target_names = labelNames, zero_division = 1))
print(confusion_matrix(yTest, yPred))

#tf.keras.utils.plot_model(model, show_shapes = True)
