# import local modules
from dictionary import COLOR_IMAGE_SIZE, GRAY_IMAGE_SIZE, LABEL_NAMES
# import foreign modules
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from numpy import argmax

def generateModelDiagram(model):
    plot_model(model, show_shapes = True)

def predictModel(model, xTest, yTest):
    # predict dataset on model
    yPred = model.predict(xTest)
    # flatten each array to highest value
    yPred = argmax(yPred, axis = 1)
    yTest = argmax(yTest, axis = 1)
    # print classification report and confusion matrix
    print(classification_report(yTest, yPred, target_names = LABEL_NAMES, zero_division = 1))
    print(confusion_matrix(yTest, yPred))

def trainModel(model, xTrain, xTest, yTrain, yTest, batchSize, iterations, verboseFlag):
    # fit model by parameters
    results = model.fit(
        xTrain, 
        yTrain, 
        validation_data = (xTest, yTest), 
        batch_size = batchSize, 
        epochs = iterations,
        verbose = verboseFlag
    )
    # evaluate model and output results
    model.evaluate(xTest, yTest)
    return model, results

def calculateCrossValidation(model, data, labels, batchSize, iterations, splits, verboseFlag):
    print("Running cross validation...")
    # set parameters for cross validation model
    crossVal_model = KerasClassifier(
        build_fn = model, 
        batch_size = batchSize, 
        epochs = iterations, 
        verbose = verboseFlag
    )
    kfold = KFold(n_splits = splits, shuffle = True, random_state = 0)
    # calculate cross validation and output results
    results = cross_val_score(crossVal_model, data, labels, cv = kfold)
    print("Cross validation results: " + str(results))
    print("%0.2f accuracy with a standard deviation of %0.2f" % (results.mean(), results.std()))

def prepareData(data, labels, trainSize):
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, train_size = trainSize, random_state = 0)
    yTrain = to_categorical(yTrain, num_classes = 10)
    yTest = to_categorical(yTest, num_classes = 10)
    return xTrain, xTest, yTrain, yTest

# Checked matplotlib for how to create two-axis graphs:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
def plot(results):
    plt.clf()
    history_dict = results.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    val_accuracy = history_dict['val_accuracy']
    epochs = range(1, (len(history_dict['loss']) + 1))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, loss_values, 'b', label = 'Training loss', c = 'lightgreen')
    ax1.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy', c = 'red')
    plt.title('Training/validation loss and accuracy')
    
    fig.tight_layout()
    plt.show()

def defineModelColor():
    #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    model = Sequential([
        Conv2D(32, (3, 3), input_shape = (COLOR_IMAGE_SIZE, COLOR_IMAGE_SIZE, 3), activation = 'relu'),
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
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = Adam(learning_rate = 0.01), 
        metrics = ['accuracy']
    )
    return model

def defineModelGray():
    model = Sequential([
        Dense(64, input_shape = (GRAY_IMAGE_SIZE, GRAY_IMAGE_SIZE), activation = 'relu'),
        Dropout(0.5),
        
        Flatten(),

        Dense(10, activation = 'sigmoid')
    ])
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = Adam(learning_rate = 0.01), 
        metrics = ['accuracy']
    )
    return model
