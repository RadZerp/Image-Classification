from datasetParser import initilizeDataset, parseDataset, segmentData, datasetIsCached, cacheDataset, loadCachedDataset
from model import prepareData, defineModel, plot
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
import logging
#remove warnings outputted by tensorflow during cross validation
tf.get_logger().setLevel(logging.ERROR)

print("\n\nGet dataset:\n\n")

if datasetIsCached():
    data, labels = loadCachedDataset()
else:
    initilizeDataset()
    images, masks, labels = parseDataset()
    data = images
    data = segmentData(images, masks)
    cacheDataset(data, labels)

print("\n\nDefine model:\n\n")

iterations = 50
batchSize = 512
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

xTrain, xTest, yTrain, yTest = prepareData(data, labels)
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
target_names = ["1","2","3","4","5","6","7","8","9","10"]
print(classification_report(yTest, yPred, target_names = target_names, zero_division = 1))
print(confusion_matrix(yTest, yPred))

#tf.keras.utils.plot_model(model, show_shapes = True)
