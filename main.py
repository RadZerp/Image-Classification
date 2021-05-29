from datasetParser import initilizeDataset, parseDataset, segmentData, datasetIsCached, cacheDataset, loadCachedDataset
from model import prepareData, defineModel, plot
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score

#get dataset

if datasetIsCached():
    data, labels = loadCachedDataset()
else:
    initilizeDataset()
    images, masks, labels = parseDataset()
    data = images
    data = segmentData(images, masks)
    cacheDataset(data, labels)

iterations = 50
batchSize = 256

#validate dataset

model = tf.keras.wrappers.scikit_learn.KerasClassifier(
    build_fn = defineModel, 
    epochs = iterations, 
    batch_size = batchSize, 
    verbose = 0
)

kfold = KFold(n_splits = 5, shuffle = True, random_state = 0) 
results = cross_val_score(model, data, labels, cv = kfold)
print("Cross validation score: " + str(results))

#train model

X_train, X_test, y_train, y_test = prepareData(data, labels)
model = defineModel()
results = model.fit(
    X_train, 
    y_train, 
    validation_data = (X_test, y_test), 
    batch_size = batchSize, 
    epochs = iterations,
    verbose = 1
)
model.evaluate(X_test, y_test)
plot(results)

#validate model

# Y_pred = model.predict(X_test)
# Y_pred = np.argmax(Y_pred, axis = 1)
# y_test = np.argmax(y_test, axis = 1)
# target_names = ["1","2","3","4","5","6","7","8","9","10"]
# print(confusion_matrix(y_test, Y_pred))
# print(classification_report(y_test, Y_pred, target_names = target_names, zero_division = 1))

#tf.keras.utils.plot_model(model, show_shapes = True)
