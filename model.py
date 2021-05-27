from sklearn import datasets
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def prepareData(data, labels): 
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = random.seed(100))
    #num_classes = len(np.unique(labels))
    #y_train = np_utils.to_categorical(y_train, num_classes)
    #y_test = np_utils.to_categorical(y_test, num_classes)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(X_test.shape)

    return X_train, X_test, y_train, y_test


from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.core import Dense, Flatten, Dropout

def defineModel():
# Tinter with model
    model = Sequential()
    model.add(Dense(5, input_shape = (None, None, 3), activation = 'relu'))
    model.add(Conv2D(3, kernel_size = (3, 3), activation = 'tanh', padding = 'valid'))# 1.1 NO 2D ONLY DENSE!
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.summary()

    return model

def plot(results):
    plt.clf()
    history_dict = results.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, (len(history_dict['loss']) + 1))
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss', c = 'lightgreen')
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    