from sklearn import datasets
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def prepareData():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = random.seed(100))
    y_train = np_utils.to_categorical(y_train, num_classes = 3)
    y_test = np_utils.to_categorical(y_test, num_classes = 3)
    return X_train, X_test, y_train, y_test


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout

def defineModel():
# Tinter with model
    model = Sequential()
    model.add(Dense(50, input_dim = 4, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
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
    
    